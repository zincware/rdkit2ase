import networkx as nx
from networkx.algorithms import isomorphism

from rdkit2ase.utils import rdkit_determine_bonds, suggestions2networkx, unwrap_molecule


def sort_templates(graphs: list[nx.Graph]) -> list[nx.Graph]:
    return sorted(
        graphs, key=lambda g: (g.number_of_nodes(), g.number_of_edges()), reverse=True
    )


def has_bond_order(graph: nx.Graph) -> bool:
    return all(x[2]["bond_order"] is not None for x in graph.edges(data=True))


def update_graph_data(
    working_graph: nx.Graph, graph: nx.Graph, mol_graph: nx.Graph
) -> bool:
    graph_matcher = isomorphism.GraphMatcher(
        working_graph,
        mol_graph,
        node_match=lambda n1, n2: n1["atomic_number"] == n2["atomic_number"],
    )
    match = next(graph_matcher.subgraph_isomorphisms_iter(), None)
    if match is None:
        return False

    # Remove matched nodes from future matching
    working_graph.remove_nodes_from(match.keys())

    # Transfer bond order from template
    inv_match = {v: k for k, v in match.items()}
    for u, v, data in mol_graph.edges(data=True):
        if "bond_order" in data:
            if data["bond_order"] is None:
                raise ValueError(
                    "Cannot update bond order with None value. "
                    "Please check the input templates."
                )
            u_g = inv_match[u]
            v_g = inv_match[v]
            if graph.has_edge(u_g, v_g):
                graph[u_g][v_g]["bond_order"] = data["bond_order"]

    # Assign charges for matched nodes
    for graph_node, template_node in match.items():
        if "charge" in mol_graph.nodes[template_node]:
            graph.nodes[graph_node]["charge"] = mol_graph.nodes[template_node]["charge"]
    return True


def update_bond_order_from_suggestions(graph, suggestions: list[nx.Graph]) -> None:
    if has_bond_order(graph):
        return

    working_graph = graph.copy()
    suggestions = sort_templates(suggestions)

    for mol_graph in suggestions:
        while True:
            updated = update_graph_data(working_graph, graph, mol_graph)
            if not updated:
                break


def update_bond_order(graph: nx.Graph, suggestions: list[str] | None = None) -> None:
    if not has_bond_order(graph):
        if suggestions is not None:
            suggestion_graphs = suggestions2networkx(suggestions)
            for component in nx.connected_components(graph):
                subgraph = graph.subgraph(component)
                update_bond_order_from_suggestions(subgraph, suggestion_graphs)
                for u, v, data in subgraph.edges(data=True):
                    graph[u][v].update(data)

        update_bond_order_determine(graph)


def update_bond_order_determine(graph: nx.Graph) -> None:
    """
    Update the bond order in the graph based on the suggestions.
    If the graph already has bond order, it will not be updated.
    """
    from rdkit2ase.networkx2x import networkx2ase
    from rdkit2ase.rdkit2x import rdkit2networkx

    if has_bond_order(graph):
        return

    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        # add pbc and cell information to the subgraph
        subgraph.graph.update(graph.graph)
        missing = sum(
            data.get("bond_order") is None for u, v, data in subgraph.edges(data=True)
        )
        # if any bond information is missing, we update the bond order
        if missing > 0:
            # Unwrapping could be made nicer, by utilizing the connectivity
            atoms = networkx2ase(subgraph)
            atoms = unwrap_molecule(atoms)
            rdkit_mol = rdkit_determine_bonds(atoms)
            rdkit_graph = rdkit2networkx(rdkit_mol)
            # update the bond order in the original graph
            update_bond_order_from_suggestions(graph, [rdkit_graph])
