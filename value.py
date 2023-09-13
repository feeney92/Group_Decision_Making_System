# The code below implements my Value-Based System. The argumentation framework can be changed
# by changing the arguments, values, attacks and supports towards the end of the code.  The value
# preferences can also be changed and/or additional users added at the end of the code.


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import itertools

# Creates a class for the arguments, containing information on their id (typically a letter), the arguments
# attacking it, the arguments supporting it, its value, its acceptance probability and its group average
# acceptance probability. NB the acceptance probability and the group average acceptance probability are
# initially set to 1 and 0 respectively, these are updated when the evaluation functions are run.
class Argument:
    def __init__(self, id):
        self.id = id
        self.attacks = set()
        self.supports = set()
        self.value = '*'
        self.accept_prob = 1
        self.group_accept_prob = 0

    # Calculate the acceptance probability of an argument for a given user's value preferences.
    def calculate_acceptance_probability(self, args, user_value_pref):
        # If an argument has no attackers it will have an acceptance probability of 1
        if self.attacks == set():
            return 1

        # Find the values that are the same as or more preferred than the value of the argument.
        all_values = get_values(args)
        arg_value = self.value

        preferred_values = []
        for value in all_values:
            if user_value_pref[value] >= user_value_pref[arg_value]:
                preferred_values.append(value)

        # Find the arguments associated with each of those values, and store them in 'pre'.
        pre = {}

        for i in range(len(preferred_values)):
            curr_value = preferred_values[i]

            temp = []
            for argument in arguments:
                if argument.value == curr_value and argument.id in self.attacks:
                    temp.append(argument.id)

                elif argument.value == curr_value and argument.id in self.supports:
                    temp.append(argument.id)

            pre[curr_value] = temp

        # Function that returns the powerset of a set.
        def powerset(s):
            power_set = []
            for r in range(len(s) + 1):
                combinations = itertools.combinations(s, r)
                power_set.extend(combinations)
            return power_set

        # Calculate the acceptance probability of the argument by calculating the probability of
        # combinations of acceptable and not acceptable arguments, and the probability that under
        # that combination the argument is defeated.
        prob_successful_attack = 0

        for value in preferred_values:
            # Consider every possible combination of arguments with a given value being accepted or not.
            # This is done by considering the power set of pre[value].  Arguments in this set represent
            # acceptable arguments associated with the given value.
            for omega in powerset(pre[value]):

                # Calculate the probability that all of the arguments in omega are acceptable.
                if len(omega) == 0:
                    acc_prob = 0

                else:
                    acc_prob = 1
                    for a in omega:
                        for argument in args:
                            if argument.id == a:
                                acc_prob *= argument.calculate_acceptance_probability(args, user_value_pref)

                # Calculate the probability that all of the arguments with the current value being considered
                # and not in omega are not acceptable.
                pre_minus_acc = set(pre[value]).difference(omega)

                if len(pre_minus_acc) == 0:
                    not_acc_prob = 1

                else:
                    not_acc_prob = 1
                    for b in pre_minus_acc:
                        for argument in args:
                            if argument.id == b:
                                not_acc_prob *= (1 - argument.calculate_acceptance_probability(args, user_value_pref))

                # # Calculate the probability that all of the arguments with a more preferred value than
                # the current value being considered are not acceptable.
                not_acc_prob_val_plus = 1

                for argument in args:
                    if argument.id in self.attacks and user_value_pref[argument.value] > user_value_pref[value]:
                        not_acc_prob_val_plus *= (1 - argument.calculate_acceptance_probability(args, user_value_pref))

                for argument in args:
                    if argument.id in self.supports and user_value_pref[argument.value] > user_value_pref[value]:
                        not_acc_prob_val_plus *= (1 - argument.calculate_acceptance_probability(args, user_value_pref))

                # Calculate the proportion of acceptable arguments that are attacks for the value being considered
                if len(omega) == 0:
                    att_prop = 0
                else:
                    att_prop = len(set(omega).intersection(self.attacks)) / len(omega)

                # Multiply all the probabilities by the proportion of acceptable attacks to calculate
                # the probability that the combination results in the argument being defeated.
                overall_prob = acc_prob * not_acc_prob * not_acc_prob_val_plus * att_prop

                # Add the probabilities for every possible combination together to determine
                # the overall total probability of a successful attack
                prob_successful_attack += overall_prob

        # The acceptance probability is then 1 - probability of a successful attack.
        return 1 - prob_successful_attack


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


# For a specified argument, returns the value associated with it.
def get_values(args):
    values = set()
    for argument in args:
        values.add(argument.value)
    return values


# Calculates the acceptance probability of all arguments for a single user.
def calculate_individual_acceptance_probability(args, user_value_pref):
    for argument in args:
        argument.accept_prob = argument.calculate_acceptance_probability(args, user_value_pref)


# Calculates the average acceptance probability of all arguments for all users.
def calculate_group_average_acceptance_probability(args, user_value_preferences):
    for user in user_value_preferences:
        calculate_individual_acceptance_probability(args, user)

        for argument in args:
            argument.group_accept_prob += argument.accept_prob / len(user_value_preferences)


# Creates a graph representation of the value bipolar argumentation framework for an individual user.
def create_individual_value_baf_graph(args):
    graph = nx.DiGraph()

    # Add arguments as nodes
    for argument in args:
        graph.add_node(argument.id, value=argument.value, accept_prob=argument.accept_prob)

    # Add attacks as directed edges
    attacks = get_attacks(args)
    for attack in attacks:
        graph.add_edge(attack[0], attack[1], linestyle='solid')

    # Add supports as directed edges
    supports = get_supports(args)
    for support in supports:
        graph.add_edge(support[0], support[1], linestyle='dotted')

    return graph


# Creates a graphical representation of the value bipolar argumentation framework for an individual user.
def visualize_individual_value_baf_graph(graph):
    pos = nx.spring_layout(graph, iterations=10000, seed=15)  # Seed can be changed for other frameworks

    edge_styles = [graph[u][v]['linestyle'] for u, v in graph.edges()]
    node_weights = list(nx.get_node_attributes(graph, 'accept_prob').values())
    node_labels = {node: f"{node}\n{(round(graph.nodes[node]['accept_prob'],2))}" for node in graph.nodes()}
    edge_labels = {(u, v): "Attack" if graph[u][v]['linestyle'] == 'solid' else "Support" for u, v in graph.edges()}

    plt.figure(figsize=(8, 6))

    # Creates a colour map for the nodes based on their acceptance probability
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
def create_group_value_baf_graph(args):
    graph = nx.DiGraph()

    # Add arguments as nodes
    for argument in args:
        graph.add_node(argument.id, group_accept_prob=argument.group_accept_prob)

    # Add attacks as directed edges
    attacks = get_attacks(args)
    for attack in attacks:
        graph.add_edge(attack[0], attack[1], linestyle='solid')

    # Add supports as directed edges
    supports = get_supports(args)
    for support in supports:
        graph.add_edge(support[0], support[1], linestyle='dotted')

    return graph


# Creates a graphical representation of the group value bipolar argumentation framework.
def visualize_group_value_baf_graph(graph):
    pos = nx.spring_layout(graph, iterations=10000, seed=15)  # Seed can be changed for other frameworks

    edge_styles = [graph[u][v]['linestyle'] for u, v in graph.edges()]
    node_weights = list(nx.get_node_attributes(graph, 'group_accept_prob').values())
    node_labels = {node: f"{node}\n{str(round(graph.nodes[node]['group_accept_prob'],2))}" for node in graph.nodes()}
    edge_labels = {(u, v): "Attack" if graph[u][v]['linestyle'] == 'solid' else "Support" for u, v in graph.edges()}

    plt.figure(figsize=(8, 6))

    # Creates a colour map for the nodes based on their acceptance probability
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


# Evaluates the group value bipolar argumentation framework and shows its graphical representation.
def evaluate_group_val_baf(args, user_value_preferences):
    calculate_group_average_acceptance_probability(args, user_value_preferences)
    graph = create_group_value_baf_graph(args)
    visualize_group_value_baf_graph(graph)


# Evaluates the value bipolar argumentation framework and shows its graphical representation for an individual user.
def evaluate_individual_val_baf(args, user_value_preferences):
    calculate_individual_acceptance_probability(args, user_value_preferences)
    graph = create_individual_value_baf_graph(args)
    visualize_individual_value_baf_graph(graph)


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

# Define the values for each argument. NB There is no need to define a value for initial claim.
arguments[1].value = 'human wellbeing'
arguments[2].value = 'trust in experts'
arguments[3].value = 'personal autonomy'
arguments[4].value = 'human wellbeing'
arguments[5].value = 'knowledge'
arguments[6].value = 'knowledge'
arguments[7].value = 'knowledge'
arguments[8].value = 'knowledge'
arguments[9].value = 'knowledge'
arguments[10].value = 'knowledge'
arguments[11].value = 'human wellbeing'
arguments[12].value = 'knowledge'
arguments[13].value = 'knowledge'
arguments[14].value = 'human wellbeing'
arguments[15].value = 'knowledge'

# Define the value preference orderings for each user (the higher the number the more preferred).
# NB the actual numbers used are not important just the relative ranking for each user.
user_value_preferences = [
    {'*': 0, 'trust in experts': 1, 'knowledge': 4, 'human wellbeing': 2, 'personal autonomy': 3},
    {'*': 0, 'trust in experts': 1, 'knowledge': 2, 'human wellbeing': 4, 'personal autonomy': 3},
    {'*': 0, 'trust in experts': 1, 'knowledge': 2, 'human wellbeing': 4, 'personal autonomy': 3},
    {'*': 0, 'trust in experts': 3, 'knowledge': 4, 'human wellbeing': 2, 'personal autonomy': 1}
    ]


evaluate_individual_val_baf(arguments, user_value_preferences[0])
evaluate_group_val_baf(arguments, user_value_preferences)
