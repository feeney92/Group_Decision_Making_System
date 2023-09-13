# The code below implements my Voting-Based System. The argumentation framework can be changed
# by changing the arguments, attacks and supports towards the end of the code.  The voting input
# can also be changed and/or additional users added at the end of the code.

# NB the voting aggregation function is the function described in Section 4. If a different categories of
# votes or a different voting function is to be used then the 'calculate_vbs' function within the
# Argument class will need to be updated.

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Creates a class for the arguments, containing information on their id (typically a letter), the arguments
# attacking it, the arguments supporting it, its vote, its vote base score and its final weight. NB the
# vote base score and final weight are initially set to 0.5, these are updated when the function
# 'evaluate_vote_baf' is run.
class Argument:
    def __init__(self, id):
        self.id = id
        self.attacks = set()
        self.supports = set()
        self.votes = []
        self.vbs = 0.5
        self.weight = 0.5

    # Calculates the vote base score for the argument based on the input user votes.
    def calculate_vbs(self):
        eps = 0.01
        if self.votes:
            weighted_score = 0
            for vote in self.votes:
                weighted_score += (vote - 1) * 0.2
            score = weighted_score / (len(self.votes) + eps)
            self.vbs = score
        else:
            self.vbs = 0.5

    # Calculates the final weight for the argument.
    def calculate_weight(self, args):
        if self.attacks == set() and self.supports == set():
            self.weight = self.vbs
            return self.weight
        else:
            E = 0
            for support in self.supports:
                for argument in args:
                    if support == argument.id:
                        E += argument.calculate_weight(args)
            for attack in self.attacks:
                for argument in args:
                    if attack == argument.id:
                        E -= argument.calculate_weight(args)
            return 1 - (1 - self.vbs**2) / (1 + self.vbs * (2 ** E))


# For a specified argument, returns the arguments that attack it.
def get_attacks(args):
    attacks = set()
    for argument in args:
        for attack in argument.attacks:
            attacks.add((attack, argument.id))
    return attacks


# For a specified argument, returns the arguments that support it.
def get_supports(args):
    supports = set()
    for argument in args:
        for support in argument.supports:
            supports.add((support, argument.id))
    return supports


# Calculates the vote base score for all arguments.
def calculate_all_vbs(args):
    for argument in args:
        argument.calculate_vbs()


# Calculates the final weight for all arguments.
def calculate_all_weights(args):
    for argument in args:
        argument.weight = argument.calculate_weight(args)


# Creates a graph representation of the voting bipolar argumentation framework.
def create_vote_baf_graph(args):
    graph = nx.DiGraph()

    # Add arguments as nodes
    for argument in args:
        graph.add_node(argument.id, weight=argument.weight)

    # Add attacks as directed edges
    attacks = get_attacks(args)
    for attack in attacks:
        graph.add_edge(attack[0], attack[1], linestyle='solid')

    # Add supports as directed edges
    supports = get_supports(args)
    for support in supports:
        graph.add_edge(support[0], support[1], linestyle='dotted')

    return graph


# Creates a graphical representation of the vote bipolar argumentation framework.
def visualize_vote_baf_graph(graph):
    pos = nx.spring_layout(graph, iterations=10000, seed=15)  # Seed can be changed for other frameworks

    edge_styles = [graph[u][v]['linestyle'] for u, v in graph.edges()]
    node_weights = list(nx.get_node_attributes(graph, 'weight').values())
    node_labels = {node: f"{node}\n{round(graph.nodes[node]['weight'],2)}" for node in graph.nodes()}
    edge_labels = {(u, v): "Attack" if graph[u][v]['linestyle'] == 'solid' else "Support" for u, v in graph.edges()}

    plt.figure(figsize=(8, 6))

    # Creates a colour map for the nodes based on their final weight
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap',
        [(0, 'red'), (0.5, 'lightgrey'), (1, 'green')],
        N=256
    )

    # Adds the nodes, node labels, edges and edge labels
    nx.draw_networkx_nodes(graph, pos, node_color=node_weights, cmap=cmap, vmin=0, vmax=1, node_size=500)
    nx.draw_networkx_labels(graph, pos, font_size=9, labels=node_labels)
    nx.draw_networkx_edges(graph, pos, edge_color='black', style=edge_styles, arrows=True, arrowsize=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black', font_size=8)

    plt.axis('on')
    plt.show()


# Evaluates the voting bipolar argumentation framework and shows its graphical representation
def evaluate_vote_baf(args):
    calculate_all_vbs(args)
    calculate_all_weights(args)
    bipolar_graph = create_vote_baf_graph(args)
    visualize_vote_baf_graph(bipolar_graph)


# Define the arguments
arguments = [
    Argument('*'),
    Argument('A'),
    Argument('B'),
    Argument('C'),
    Argument('D'),
    Argument('E'),
    Argument('F'),
    Argument('G'),
    Argument('H'),
    Argument('I'),
    Argument('J'),
    Argument('K'),
    Argument('L'),
    Argument('M'),
    Argument('N'),
    Argument('O'),
]

# Define the attacks
arguments[0].attacks = ['A', 'L']
arguments[1].attacks = ['B', 'C']
arguments[6].attacks = ['I']
arguments[10].attacks = ['K']
arguments[12].attacks = ['O']

# Define the supports
arguments[0].supports = ['E', 'F', 'J']
arguments[1].supports = ['D']
arguments[6].supports = ['H', 'G']
arguments[12].supports = ['M', 'N']

# Provide user voting input for each argument.  NB the system assumes the same number of votes
# are provided for each argument.
arguments[1].votes = [2, 6, 5, 3]
arguments[2].votes = [6, 4, 4, 4]
arguments[3].votes = [5, 5, 2, 2]
arguments[4].votes = [2, 3, 5, 4]
arguments[5].votes = [3, 1, 1, 1]
arguments[6].votes = [1, 1, 2, 2]
arguments[7].votes = [1, 3, 2, 2]
arguments[8].votes = [2, 2, 3, 3]
arguments[9].votes = [4, 5, 5, 3]
arguments[10].votes = [4, 4, 3, 5]
arguments[11].votes = [6, 4, 6, 2]
arguments[12].votes = [1, 2, 4, 6]
arguments[13].votes = [2, 2, 5, 5]
arguments[14].votes = [5, 2, 4, 6]
arguments[15].votes = [2, 3, 5, 2]

evaluate_vote_baf(arguments)
