import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
from heapdict import heapdict

class Node:
    def __init__(self, state, parent=None, action=None, totalCost: float = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.totalCost = totalCost

    def __lt__(self, other):
        return (self.totalCost, self.state) < (other.totalCost, other.state)

class DFSGAgent:
    def __init__(self):
        self.env = None
        self.open = [] 
        self.close = []  
        self.totalExpended = 0

    def search(self, env: 'CampusEnv') -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.open = []  
        self.close = []
        self.totalExpended = 0
        initial_state = self.env.get_state()
        node = Node(initial_state, parent=None, action=None, totalCost=0)
        
        self.open.append(node)
        
        result = self.recursive_dfs(env)
        if result is not None:
            actions, totalCost = result
            return actions, totalCost, self.totalExpended
        else:
            return [], float('inf'), self.totalExpended
    
    def recursive_dfs(self, env: 'CampusEnv') -> Tuple[List[int], float, int]:
        if not self.open:
            return None
        node = self.open.pop()  # Get the deepest node (LIFO)
        self.close.append(node.state)
        
        if self.env.is_final_state(node.state):
            actions = []
            tmpNode = node
            while tmpNode.parent is not None:
                actions.append(tmpNode.action)
                tmpNode = tmpNode.parent
            actions.reverse()
            return actions, node.totalCost
        
        for action, successor in env.succ(node.state).items():
            if successor[0] is node.state or successor[0] == None:
                continue
            child_state = successor[0]
            cost = successor[1]
            if child_state not in self.close and not any(n.state == child_state for n in self.open):
                self.totalExpended += 1
                child_node = Node(child_state, parent=node, action=action, totalCost=node.totalCost + cost)
                self.open.append(child_node)
                result = self.recursive_dfs(env)
                
                if result is not None:
                    return result
        
        return None

  
class UCSAgent():
    def __init__(self):
        self.env = None
        self.open = heapdict()  # Priority queue for UCS
        self.close = set()  # Set to keep track of explored states

    def search(self, env: 'CampusEnv') -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.open = heapdict()
        self.close = set()
        totalExpended = 0
        initial_state = self.env.get_state()
        node = Node(initial_state, parent=None, action=None, totalCost=0)
        
        self.open[node] = (node.totalCost, node.state)
        
        while self.open:
            node, _ = self.open.popitem()

            env.set_state(node.state)
            self.close.add(node.state)
            
            if self.env.is_final_state(node.state):
                actions = []
                tmpNode = node
                while tmpNode.parent is not None:
                    actions.append(tmpNode.action)
                    tmpNode = tmpNode.parent
                actions.reverse()
                return actions, node.totalCost, totalExpended
            totalExpended += 1
            for action, successor in env.succ(node.state).items():
                if successor[0] is node.state or successor[0] is None:
                    continue
                
                child_state = successor[0]
                cost = successor[1]
                new_cost = node.totalCost + cost
                
                if child_state not in self.close and child_state not in [n.state for n in self.open]:
                    child_node = Node(child_state, parent=node, action=action, totalCost=new_cost)
                    self.open[child_node] = (new_cost, child_state)
                elif child_state in [n.state for n in self.open]:
                    for n in self.open:
                        if n.state == child_state and n.totalCost > new_cost:
                            del self.open[n]
                            child_node = Node(child_state, parent=node, action=action, totalCost=new_cost)
                            self.open[child_node] = (new_cost, child_state)
                            break
        
        return [], float('inf'), totalExpended


class A_Node:
    def __init__(self, state, parent=None, action=None, g: float = 0, h: float = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # Cost to reach this node
        self.h = h  # Heuristic value

class WeightedAStarAgent():
    def HCampus(self, state):
        min = 100
        row, col = self.env.to_row_col(state);
        for goal in self.env.goals:
            g_row, g_col = self.env.to_row_col(goal);
            dist = abs(row - g_row) + abs(col - g_col)
            if dist < min:
                min = dist
        return min

        

    def __init__(self):
        self.env = None
        self.open = heapdict() 
        self.close = set()  

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        self.open = heapdict() 
        self.close = set()  
        totalExpended = 0
        initial_state = self.env.get_state()
        initial_h = self.HCampus(initial_state) 
        node = A_Node(initial_state, parent=None, action=None, g=0, h=initial_h)
        
        self.open[node] = (node.h * h_weight + node.g *(1-h_weight), node.state)
        
        while self.open:
            node, _ = self.open.popitem()
            
            if self.env.is_final_state(node.state):
                actions = []
                tmpNode = node
                while tmpNode.parent is not None:
                    actions.append(tmpNode.action)
                    tmpNode = tmpNode.parent
                actions.reverse()
                return actions, node.g, totalExpended
            
            self.close.add(node.state)
            totalExpended += 1
            
            for action, successor in env.succ(node.state).items():
                child_state = successor[0]
                if child_state is node.state or child_state is None:
                    continue
                cost = successor[1]
                new_g = node.g + cost
                new_h = self.HCampus(child_state)
                new_f = new_h * h_weight + new_g *(1-h_weight)
                
                if child_state not in self.close:
                    if any(n.state == child_state for n in self.open):
                        for n in self.open:
                            if n.state == child_state and n.g > new_g:
                                del self.open[n]
                                child_node = A_Node(child_state, parent=node, action=action, g=new_g, h=new_h)
                                self.open[child_node] = (new_f, child_state)
                                break
                    else:
                        child_node = A_Node(child_state, parent=node, action=action, g=new_g, h=new_h)
                        self.open[child_node] = (new_f, child_state)
        
        return [], float('inf'), totalExpended



class AStarAgent():
    
    def __init__(self):
        self.weighted_agent = WeightedAStarAgent()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return self.weighted_agent.search(env, h_weight=0.5)

