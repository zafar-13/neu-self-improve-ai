import math
import random

class TreeNode:
    """A node in the Monte Carlo Search Tree."""
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

def MCTS_Search(root_state, num_iterations):
    """
    Performs the MCTS search and returns the best move.
    """
    root_node = TreeNode(state=root_state)

    for _ in range(num_iterations):
        # 1. Selection
        leaf = select(root_node)
        
        # 2. Expansion
        expand(leaf)
        
        # 3. Simulation
        node_to_simulate = random.choice(leaf.children) if leaf.children else leaf
        result = simulate(node_to_simulate)
        
        # 4. Backpropagation
        backpropagate(node_to_simulate, result)
        
    # After the loop, choose the best move from the root
    best_move_node = choose_best_move(root_node)
    return best_move_node.state.last_action

def select(node):
    """
    Finds the best node to expand by repeatedly applying UCB1.
    """
    while node.children:
        unvisited_children = [child for child in node.children if child.visits == 0]
        if unvisited_children:
            return random.choice(unvisited_children)
        
        node = find_best_child_with_UCB1(node)
    return node

def find_best_child_with_UCB1(node, exploration_constant=1.41):
    """
    Selects the best child using the UCB1 formula.
    """
    best_score = -1
    best_child = None
    for child in node.children:
        exploit_score = child.wins / child.visits
        explore_score = math.sqrt(math.log(node.visits) / child.visits)
        ucb_score = exploit_score + exploration_constant * explore_score
        
        if ucb_score > best_score:
            best_score = ucb_score
            best_child = child
    return best_child

def expand(node):
    """
    Creates all possible child nodes for a given node.
    """
    if not node.state.is_terminal():
        possible_moves = node.state.get_legal_actions()
        for move in possible_moves:
            new_state = node.state.move(move)
            child_node = TreeNode(state=new_state, parent=node)
            node.children.append(child_node)

def simulate(node):
    """
    Runs a random playout from a node's state to the end of the game.
    """
    current_state = node.state
    while not current_state.is_terminal():
        random_move = random.choice(current_state.get_legal_actions())
        current_state = current_state.move(random_move)
    
    return current_state.get_game_result(node.parent.state.player_turn)

def backpropagate(node, result):
    """
    Updates the visit and win counts from the leaf up to the root.
    """
    temp_node = node
    while temp_node is not None:
        temp_node.visits += 1

        if temp_node.parent and temp_node.parent.state.player_turn != temp_node.state.player_turn:
             result = 1 - result 
        temp_node.wins += result
        temp_node = temp_node.parent

def choose_best_move(root_node):
    """
    Selects the child of the root with the highest number of visits.
    """
    most_visited_child = max(root_node.children, key=lambda child: child.visits)
    return most_visited_child