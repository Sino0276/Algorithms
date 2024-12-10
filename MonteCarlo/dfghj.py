import numpy as np
import random

# Tic-Tac-Toe Board Class
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 보드 초기화 (0: 빈칸, 1: 플레이어 1, -1: 플레이어 2)
        self.current_player = 1  # 현재 플레이어 (1 또는 -1)

    def reset(self):
        """게임 보드를 초기화합니다."""
        self.board.fill(0)
        self.current_player = 1

    def get_legal_moves(self):
        """가능한 움직임(빈 칸)을 반환합니다."""
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def make_move(self, move):
        """현재 플레이어가 움직임을 수행합니다."""
        r, c = move
        if self.board[r, c] == 0:
            self.board[r, c] = self.current_player
            self.current_player *= -1  # 플레이어 전환
            return True
        return False

    def check_winner(self):
        """게임 상태를 확인합니다. 1, -1 (승자) 또는 0 (진행 중/무승부)을 반환합니다."""
        for i in range(3):
            # 행 또는 열에서 승리 확인
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return np.sign(sum(self.board[i, :])) or np.sign(sum(self.board[:, i]))
        # 대각선에서 승리 확인
        if abs(sum(self.board.diagonal())) == 3 or abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return np.sign(sum(self.board.diagonal())) or np.sign(sum(np.fliplr(self.board).diagonal()))
        # 무승부 또는 진행 중
        if not self.get_legal_moves():
            return 0  # 무승부
        return None  # 게임 진행 중

    def display(self):
        """현재 보드 상태를 출력합니다."""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print()

# MCTS Node Class
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # 현재 상태
        self.parent = parent  # 부모 노드
        self.children = []  # 자식 노드
        self.visits = 0  # 방문 횟수
        self.wins = 0  # 승리 횟수
        self.move = move  # 이 노드로 오는 데 사용된 움직임

    def is_fully_expanded(self):
        """노드가 완전히 확장되었는지 확인합니다."""
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.0):
        """UCT 값을 기준으로 가장 좋은 자식 노드를 반환합니다."""
        return max(self.children, key=lambda child: (child.wins / child.visits) +
                   exploration_weight * np.sqrt(np.log(self.visits) / child.visits))

    def expand(self):
        """현재 상태에서 가능한 움직임으로 새 노드를 확장합니다."""
        legal_moves = self.state.get_legal_moves()
        for move in legal_moves:
            if all(child.move != move for child in self.children):  # 중복 방지
                new_state = TicTacToe()
                new_state.board = np.copy(self.state.board)
                new_state.current_player = self.state.current_player
                new_state.make_move(move)
                child_node = MCTSNode(new_state, parent=self, move=move)
                self.children.append(child_node)
                return child_node
        return None

    def backpropagate(self, result):
        """결과를 현재 노드와 부모로 역전파합니다."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)

    def simulate(self):
        """현재 상태에서 무작위로 플레이해 결과를 반환합니다."""
        temp_state = TicTacToe()
        temp_state.board = np.copy(self.state.board)
        temp_state.current_player = self.state.current_player
        while temp_state.check_winner() is None:
            move = random.choice(temp_state.get_legal_moves())
            temp_state.make_move(move)
        return temp_state.check_winner()

# MCTS Algorithm
def mcts_search(root, iterations=1000):
    """
    MCTS 알고리즘을 실행하여 최적의 움직임을 반환합니다.
    Parameters:
        root (MCTSNode): 루트 노드 (현재 상태를 포함).
        iterations (int): 실행할 반복 횟수.
    Returns:
        MCTSNode: 선택된 최적의 자식 노드.
    """
    for _ in range(iterations):
        node = root
        # 1. 선택 단계
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        # 2. 확장 단계
        if not node.is_fully_expanded():
            node = node.expand()
        # 3. 시뮬레이션 단계
        result = node.simulate()
        # 4. 역전파 단계
        node.backpropagate(result)
    return root.best_child(exploration_weight=0)  # 탐색 가중치 제거하여 최적 선택

# 테스트: Tic-Tac-Toe 게임에서 AI 실행
game = TicTacToe()
root = MCTSNode(game)

# MCTS를 사용해 최적 움직임 선택
while game.check_winner() is None:
    game.display()
    if game.current_player == 1:  # AI 턴
        root = MCTSNode(game)
        best_move_node = mcts_search(root, iterations=1000)
        game.make_move(best_move_node.move)
    else:  # 플레이어 (랜덤) 턴
        move = random.choice(game.get_legal_moves())
        game.make_move(move)

# 최종 결과 출력
game.display()
winner = game.check_winner()
if winner == 1:
    print("AI 승리!")
elif winner == -1:
    print("플레이어 승리!")
else:
    print("무승부!")
