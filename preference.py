# The code below implements my Preference-Based System. The argumentation framework can be changed
# by changing the arguments, values, attacks and supports towards the end of the code.  The argument
# preferences can also be changed and/or additional users added at the end of the code.


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Creates a class for the arguments, containing information on their id (typically a letter), the arguments
# attacking it, the arguments supporting it, where it is accepted or not and the proportion of users
# for which it is accepted. NB the acceptance the group acceptance proportion are  initially set to 1
# and 0 respectively, these are updated when the evaluation functions are run.


class Argument:
    def __init__(self, id):
        self.id = id
        self.attacks = set()
        self.supports = set()
        self.acceptance = 1
        self.group_acceptance_prop = 0

    # Calculate if an argument is accepted for a given user's preferences. A value of 1 means the argument
    # is accepted, and 0 otherwise.
    def calculate_acceptance(self, args, user_pref):
        # If an argument has no attackers it will be accepted
        if self.attacks == set():
            return 1

        # If an argument has an attacker which is more preferred it will not be accepted
        elif self.attacks != set() and self.supports == set():
            for attack in self.attacks:
                for argument in args:
                    if attack == argument.id:
                        if user_pref[argument.id] >= user_pref[self.id]:
                            return 0
            return 1

        # If an argument has an supporter which is more preferred than any attacker then the argument
        # will be accepted
        else:
            most_pref_arg = self.id
            for argument in self.attacks + self.supports:
                for arg in args:
                    if argument == arg.id:
                        if user_pref[arg.id] > user_pref[most_pref_arg] and arg.calculate_acceptance(args, user_pref) > 0:
                            most_pref_arg = arg.id
            if most_pref_arg in self.supports or most_pref_arg == self.id:
                return 1
            else:
                return 0


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


# Calculates whether each of the arguments is accepted or not for a single user.
def calculate_individual_acceptance(args, user_pref):
    for argument in args:
        argument.acceptance = argument.calculate_acceptance(args, user_pref)


# For each argument, calculates the proportion of users that find it acceptable.
def calculate_group_average_acceptance(args, user_preferences):
    for user in user_preferences:
        calculate_individual_acceptance(args, user)

        for argument in args:
            argument.group_acceptance_prop += argument.acceptance / len(user_preferences)


# Creates a graph representation of the preference bipolar argumentation framework for an individual user.
def create_individual_pref_baf_graph(args):
    graph = nx.DiGraph()

    # Add arguments as nodes
    for argument in args:
        graph.add_node(argument.id, acceptance=argument.acceptance)

    # Add attacks as directed edges
    attacks = get_attacks(args)
    for attack in attacks:
        graph.add_edge(attack[0], attack[1], linestyle='solid')

    # Add supports as directed edges
    supports = get_supports(args)
    for support in supports:
        graph.add_edge(support[0], support[1], linestyle='dotted')

    return graph


# Creates a graphical representation of the preference bipolar argumentation framework for an individual user.
def visualize_individual_pref_baf_graph(graph):
    pos = nx.spring_layout(graph, iterations=10000, seed=15) # Seed can be changed for other frameworks

    edge_styles = [graph[u][v]['linestyle'] for u, v in graph.edges()]
    node_weights = list(nx.get_node_attributes(graph, 'acceptance').values())
    node_labels = {node: f"{node}" for node in graph.nodes()}
    edge_labels = {(u, v): "Attack" if graph[u][v]['linestyle'] == 'solid' else "Support" for u, v in graph.edges()}

    plt.figure(figsize=(8, 6))

    # Creates a colour map for the nodes based on whether they are accepted or not
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap',
        [(0, 'red'), (0.5, 'lightgrey'), (1, 'green')],
        N=256
    )

    # Adds the nodes, node label, edges and edge labels
    nx.draw_networkx_nodes(graph, pos, node_color=node_weights, cmap=cmap, vmin=0, vmax=1, node_size=500)
    nx.draw_networkx_labels(graph, pos, font_size=9, labels=node_labels)
    nx.draw_networkx_edges(graph, pos, edge_color='black', style=edge_styles, arrows=True, arrowsize=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black', font_size=8)

    plt.axis('on')
    plt.show()


# Creates a graph representation of the group value bipolar argumentation framework.
def create_group_pref_baf_graph(args):
    graph = nx.DiGraph()

    # Add arguments as nodes
    for argument in args:
        graph.add_node(argument.id, group_acceptance_prop=argument.group_acceptance_prop)

    # Add attacks as directed edges
    attacks = get_attacks(args)
    for attack in attacks:
        graph.add_edge(attack[0], attack[1], linestyle='solid')

    # Add supports as directed edges
    supports = get_supports(args)
    for support in supports:
        graph.add_edge(support[0], support[1], linestyle='dotted')

    return graph


# Creates a graphical representation of the group preference bipolar argumentation framework.
def visualize_group_pref_baf_graph(graph):
    pos = nx.spring_layout(graph, iterations=10000, seed=15) # Seed can be changed for other frameworks

    edge_styles = [graph[u][v]['linestyle'] for u, v in graph.edges()]
    node_weights = list(nx.get_node_attributes(graph, 'group_acceptance_prop').values())
    node_labels = {node: f"{node}\n{str(round(graph.nodes[node]['group_acceptance_prop'],2))}" for node in graph.nodes()}
    edge_labels = {(u, v): "Attack" if graph[u][v]['linestyle'] == 'solid' else "Support" for u, v in graph.edges()}

    plt.figure(figsize=(8, 6))

    # Creates a colour map for the nodes based on whether they are accepted or not.
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


# Evaluates the group preference bipolar argumentation framework and shows its graphical representation.
def evaluate_group_pref_baf(args, user_preferences):
    calculate_group_average_acceptance(args, user_preferences)
    graph = create_group_pref_baf_graph(args)
    visualize_group_pref_baf_graph(graph)


# Evaluates the preference bipolar argumentation framework and shows its graphical representation for an individual
# user.
def evaluate_individual_pref_baf(args, user_preferences):
    calculate_individual_acceptance(args, user_preferences)
    graph = create_individual_pref_baf_graph(args)
    visualize_individual_pref_baf_graph(graph)


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


# Define the argument preference orderings for each user (the higher the number the more preferred).
# NB the actual numbers used are not important just the relative ranking for each user.
user_preferences = [
    {'*': 0, 'A': 3, 'B': 6, 'C': 5, 'D': 4, 'E': 4, 'F': 2, 'G': 3, 'H': 4, 'I': 5,
    'J': 5, 'K': 6, 'L': 1, 'M': 3, 'N': 4, 'O': 2},

    {'*': 0, 'A': 5, 'B': 3, 'C': 4, 'D': 2, 'E': 1, 'F': 2, 'G': 4, 'H': 3, 'I': 5,
    'J': 4, 'K': 5, 'L': 3, 'M': 2, 'N': 4, 'O': 5},

    {'*': 0, 'A': 3, 'B': 5, 'C': 2, 'D': 4, 'E': 1, 'F': 2, 'G': 3, 'H': 5, 'I': 4,
     'J': 4, 'K': 3, 'L': 5, 'M': 3, 'N': 4, 'O': 2},

    {'*': 0, 'A': 5, 'B': 3, 'C': 2, 'D': 4, 'E': 1, 'F': 1, 'G': 2, 'H': 3, 'I': 4,
    'J': 3, 'K': 4, 'L': 4, 'M': 6, 'N': 3, 'O': 5}
    ]

evaluate_individual_pref_baf(arguments, user_preferences[0])
evaluate_group_pref_baf(arguments, user_preferences)

