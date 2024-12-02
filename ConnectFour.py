import numpy as np
import pygame
import sys

ROWS = 6
COLS = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PLAYER_COLOR = None
AI_COLOR = None

# Create the game board
def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

# Check if a move is valid
def is_valid_move(board, col):
    return board[0][col] == 0

# Check if the board is full
def is_full(board):
    return all(board[0][c] != 0 for c in range(COLS))  # If the top row is full, the board is full

# Drop a piece into the board
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Get the next open row in a column
def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r

# Heuristic evaluation of the board for the AI
def evaluate_window(window, piece):
    score = 0
    opponent_piece = 1 if piece == -1 else -1  # Opponent logic adjusted
    if window.count(piece) == 4:
        score += 10  # Win condition
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 6  # Strong 
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 3  # Weak 
    if window.count(opponent_piece) == 1 and window.count(0) == 3:
        score -= 1     
    elif window.count(opponent_piece) == 3 and window.count(0) == 1:
        score -= 4  # Block opponent's strong winning chance
    if window.count(opponent_piece) == 2 and window.count(0) == 2:
        score -= 2  # Block opponent's weak winning chance
    return score
def score_position(board, piece):
    score = 0

    # Center column preference
    center_array = [int(board[row][COLS // 2]) for row in range(ROWS)]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Horizontal score
    for row in range(ROWS):
        row_array = [int(board[row][col]) for col in range(COLS)]
        for col in range(COLS - 3):
            window = row_array[col:col + 4]
            score += evaluate_window(window, piece)

    # Vertical score
    for col in range(COLS):
        col_array = [int(board[row][col]) for row in range(ROWS)]
        for row in range(ROWS - 3):
            window = col_array[row:row + 4]
            score += evaluate_window(window, piece)

    # Positive diagonal score
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            window = [board[row + i][col + i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Negative diagonal score
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            window = [board[row - i][col + i] for i in range(4)]
            score += evaluate_window(window, piece)
    return score

def evaluate_board(board):
#   Heuristic evaluation of the board for the AI.
    ai_score = score_position(board, -1)  # AI's piece is -1
    player_score = score_position(board, 1)  # Player's piece is 1
    return ai_score - player_score  # AI tries to maximize this score

# Maximizing function for Minimax
def maximize(state, depth):
    if depth == 0 or is_full(state):  # Terminal condition
        evaluation = evaluate_board(state)
        return None, evaluation
    
    max_child, max_utility = None, -float('inf')
    valid_moves = [c for c in range(COLS) if is_valid_move(state, c)]
    
    for col in valid_moves:
        row = get_next_open_row(state, col)
        child_state = state.copy()
        drop_piece(child_state, row, col, -1)  # AI's move (AI = -1)
        
        _, utility = minimize(child_state, depth - 1)  # Switch to minimizing
        if utility > max_utility:
            max_child, max_utility = col, utility
    
    return max_child, max_utility

# Minimizing function for Minimax 
def minimize(state, depth):
    if depth == 0 or is_full(state):  # Terminal condition
        evaluation = evaluate_board(state)
        return None, evaluation
    
    min_child, min_utility = None, float('inf')
    valid_moves = [c for c in range(COLS) if is_valid_move(state, c)]
    
    for col in valid_moves:
        row = get_next_open_row(state, col)
        child_state = state.copy()
        drop_piece(child_state, row, col, 1)  # Player's move (Player = 1)
        
        _, utility = maximize(child_state, depth - 1)  # Switch to maximizing
        if utility < min_utility:
            min_child, min_utility = col, utility
    
    return min_child, min_utility

def maximize_with_pruning(state, depth, alpha, beta):
    if depth == 0 or is_full(state):  # Terminal condition
        evaluation = evaluate_board(state)
        return None, evaluation
    
    max_child, max_utility = None, -float('inf')
    valid_moves = [c for c in range(COLS) if is_valid_move(state, c)]
    
    for col in valid_moves:
        row = get_next_open_row(state, col)
        child_state = state.copy()
        drop_piece(child_state, row, col, -1)  # AI's move (AI = -1)
        
        _, utility = minimize_with_pruning(child_state, depth - 1, alpha, beta)  # Switch to minimizing
        
        if utility > max_utility:
            max_child, max_utility = col, utility
        
        # Alpha-Beta Pruning
        alpha = max(alpha, max_utility)
        if beta <= alpha:
            break  # Prune the remaining branches
    
    return max_child, max_utility

# Minimizing function for Minimax with Alpha-Beta Pruning
def minimize_with_pruning(state, depth, alpha, beta):
    if depth == 0 or is_full(state):  # Terminal condition
        evaluation = evaluate_board(state)
        return None, evaluation
    
    min_child, min_utility = None, float('inf')
    valid_moves = [c for c in range(COLS) if is_valid_move(state, c)]
    
    for col in valid_moves:
        row = get_next_open_row(state, col)
        child_state = state.copy()
        drop_piece(child_state, row, col, 1)  # Player's move (Player = 1)
        
        _, utility = maximize_with_pruning(child_state, depth - 1, alpha, beta)  # Switch to maximizing
        
        if utility < min_utility:
            min_child, min_utility = col, utility
        
        # Alpha-Beta Pruning
        beta = min(beta, min_utility)
        if beta <= alpha:
            break  # Prune the remaining branches
    
    return min_child, min_utility
def expect_maximize(state, depth):
    if depth == 0 or is_full(state):  # Terminal condition
        evaluation = evaluate_board(state)
        return None, evaluation
    
    max_child = None
    max_utility = -float('inf')
    valid_moves = [c for c in range(COLS) if is_valid_move(state, c)]
    
    for col in valid_moves:
        row = get_next_open_row(state, col)
        child_state = state.copy()
        drop_piece(child_state, row, col, -1)  # AI's move (AI = -1)

        expected_utility = 0
        probabilities = [0.6, 0.2, 0.2]
        offsets = [0, -1, 1]  # Current column, left, right
        # Calculate expected utility based on probabilities of moving left, right, or staying in the current column
        for i in range(len(probabilities)):
            offset = offsets[i]
            prob = probabilities[i]
            neighbor_col = col + offset
            if 0 <= neighbor_col < COLS and is_valid_move(state, neighbor_col):
                neighbor_row = get_next_open_row(state, neighbor_col)
                neighbor_state = state.copy()
                drop_piece(neighbor_state, neighbor_row, neighbor_col, -1)
                _, utility = expect_minimize(neighbor_state, depth - 1)
                expected_utility += prob * utility
            else:
                expected_utility += prob * evaluate_board(state)  # Stay at current evaluation for invalid moves

        if expected_utility > max_utility:
            max_utility = expected_utility
            max_child = col

    return max_child, max_utility


def expect_minimize(state, depth):
    if depth == 0 or is_full(state):  # Terminal condition
        evaluation = evaluate_board(state)
        return None, evaluation
    
    min_child = None
    min_utility = float('inf')
    valid_moves = [c for c in range(COLS) if is_valid_move(state, c)]
    
    for col in valid_moves:
        row = get_next_open_row(state, col)
        child_state = state.copy()
        drop_piece(child_state, row, col, 1)  # Player's move (Player = 1)

        expected_utility = 0
        probabilities = [0.6, 0.2, 0.2]
        offsets = [0, -1, 1]  
        
        for i in range(len(probabilities)):
            offset = offsets[i]
            prob = probabilities[i]
            neighbor_col = col + offset
            if 0 <= neighbor_col < COLS and is_valid_move(state, neighbor_col):
                neighbor_row = get_next_open_row(state, neighbor_col)
                neighbor_state = state.copy()
                drop_piece(neighbor_state, neighbor_row, neighbor_col, 1)
                _, utility = expect_maximize(neighbor_state, depth - 1)
                expected_utility += prob * utility
            else:
                expected_utility += prob * evaluate_board(state)  

        if expected_utility < min_utility:
            min_utility = expected_utility
            min_child = col

    return min_child, min_utility

# AI move selection based on Minimax
def ai_move(board, depth):
    col, _ = maximize(board, depth)  # AI is the maximizer
    return col
def ai_move_with_pruning(board, depth):
    col, _ = maximize_with_pruning(board, depth, -float('inf'), float('inf'))  # AI is the maximizer
    return col
def ai_move_expectimax(board, depth):
    col, _ = expect_maximize(board, depth)   # AI is the maximizer
    return col
# Draw the board
def draw_board(board, screen):
    for r in range(ROWS):
        for c in range(COLS):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (c * SQUARESIZE + SQUARESIZE // 2, r * SQUARESIZE + SQUARESIZE + SQUARESIZE // 2), RADIUS)

    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == 1:
                pygame.draw.circle(screen, PLAYER_COLOR, (c * SQUARESIZE + SQUARESIZE // 2, (r + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)
            elif board[r][c] == -1:
                pygame.draw.circle(screen, AI_COLOR, (c * SQUARESIZE + SQUARESIZE // 2, (r + 1) * SQUARESIZE + SQUARESIZE // 2), RADIUS)
    pygame.display.update()

def calculate_final_scores(board):
    player_score = 0
    ai_score = 0

    # Set to track already counted positions
    counted_positions = set()

    # Horizontal check
    for row in range(ROWS):
        for col in range(COLS - 3):  # Ensure 4 cells fit horizontally
            window = [board[row][col + i] for i in range(4)]
            positions = [(row, col + i) for i in range(4)]
            if set(positions).isdisjoint(counted_positions):  # Only count if not already counted
                if window == [1, 1, 1, 1]:  # Player's piece
                    player_score += 1
                    print(f"Player 4-in-row horizontally at: {positions}")
                    counted_positions.update(positions)  # Add positions to counted
                elif window == [-1, -1, -1, -1]:  # AI's piece
                    ai_score += 1
                    print(f"AI 4-in-row horizontally at: {positions}")
                    counted_positions.update(positions)  # Add positions to counted

    # Vertical check
    for col in range(COLS):
        for row in range(ROWS - 3):  # Ensure 4 cells fit vertically
            window = [board[row + i][col] for i in range(4)]
            positions = [(row + i, col) for i in range(4)]
            if set(positions).isdisjoint(counted_positions):  # Only count if not already counted
                if window == [1, 1, 1, 1]:  # Player's piece
                    player_score += 1
                    print(f"Player 4-in-row vertically at: {positions}")
                    counted_positions.update(positions)  # Add positions to counted
                elif window == [-1, -1, -1, -1]:  # AI's piece
                    ai_score += 1
                    print(f"AI 4-in-row vertically at: {positions}")
                    counted_positions.update(positions)  # Add positions to counted

    # Positive diagonal check (/)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            window = [board[row + i][col + i] for i in range(4)]
            positions = [(row + i, col + i) for i in range(4)]
            if set(positions).isdisjoint(counted_positions):  # Only count if not already counted
                if window == [1, 1, 1, 1]:  # Player's piece
                    player_score += 1
                    print(f"Player 4-in-row diagonally (\\) at: {positions}")
    # negative diagonal check (\)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            window = [board[row - i][col + i] for i in range(4)]
            positions = [(row - i, col + i) for i in range(4)]
            if set(positions).isdisjoint(counted_positions):  # Only count if not already counted
                if window == [1, 1, 1, 1]:  # Player's piece
                    player_score += 1
                    print(f"Player 4-in-row diagonally (/) at: {positions}")                
    print(f"Final Scores")
    print(f"Player Score: {player_score}")
    print(f"AI Score: {ai_score}")        


# Main game loop
def setup_screen():
    pygame.init()
    screen = pygame.display.set_mode((500, 600))  # Adjusted size to fit all elements
    pygame.display.set_caption("Connect Four Setup")
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 28)
    
    clock = pygame.time.Clock()

    player_color = None
    ai_color = None
    depth = ""
    input_active = False
    use_alpha_beta = False  # Default to no Alpha-Beta pruning
    use_expectimax = False  # Default to not using Expectimax
    running = True

    while running:
        screen.fill((255, 255, 255))  # White background
        
        # Display title
        title_text = font.render("Connect Four Setup", True, (0, 0, 0))
        screen.blit(title_text, (130, 20))

        # Color choices
        red_button = pygame.Rect(50, 100, 150, 50)
        green_button = pygame.Rect(300, 100, 150, 50)
        pygame.draw.rect(screen, RED, red_button)
        pygame.draw.rect(screen, GREEN, green_button)

        red_text = small_font.render("Red (Player)", True, (255, 255, 255))
        green_text = small_font.render("Green (Player)", True, (255, 255, 255))
        screen.blit(red_text, (70, 115))
        screen.blit(green_text, (310, 115))

        # Depth input field
        depth_text = font.render("Enter AI Depth:", True, (0, 0, 0))
        screen.blit(depth_text, (150, 200))

        input_box = pygame.Rect(150, 250, 200, 50)
        pygame.draw.rect(screen, (200, 200, 200), input_box)
        depth_display = font.render(depth, True, (0, 0, 0))
        screen.blit(depth_display, (160, 260))

        # Alpha-Beta pruning options
        alpha_beta_button = pygame.Rect(50, 320, 200, 50)
        no_alpha_beta_button = pygame.Rect(250, 320, 200, 50)
        pygame.draw.rect(screen, (0, 0, 255), alpha_beta_button)
        pygame.draw.rect(screen, (255, 0, 0), no_alpha_beta_button)

        alpha_beta_text = small_font.render("With Alpha-Beta", True, (255, 255, 255))
        no_alpha_beta_text = small_font.render("Without Alpha-Beta", True, (255, 255, 255))
        screen.blit(alpha_beta_text, (90, 335))
        screen.blit(no_alpha_beta_text, (270, 335))

        # Expectimax option
        expectimax_button = pygame.Rect(50, 390, 400, 50)
        pygame.draw.rect(screen, (128, 0, 128), expectimax_button)
        expectimax_text = small_font.render("Use Expectiminimax Algorithm", True, (255, 255, 255))
        screen.blit(expectimax_text, (70, 405))

        # Start button
        start_button = pygame.Rect(200, 450, 100, 50)
        pygame.draw.rect(screen, (0, 0, 255), start_button)
        start_text = small_font.render("Start", True, (255, 255, 255))
        screen.blit(start_text, (230, 465))
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                # Select player color
                if red_button.collidepoint(mouse_pos):
                    player_color = RED
                    ai_color = GREEN
                elif green_button.collidepoint(mouse_pos):
                    player_color = GREEN
                    ai_color = RED
                
                # Check if depth input box is clicked
                if input_box.collidepoint(mouse_pos):
                    input_active = True
                else:
                    input_active = False
                
                # Toggle Alpha-Beta pruning
                if alpha_beta_button.collidepoint(mouse_pos):
                    use_alpha_beta = True
                    use_expectimax = False  # Disable Expectimax
                elif no_alpha_beta_button.collidepoint(mouse_pos):
                    use_alpha_beta = False
                    use_expectimax = False  # Disable Expectimax
                
                # Toggle Expectimax
                if expectimax_button.collidepoint(mouse_pos):
                    use_expectimax = True
                    use_alpha_beta = False  # Disable Alpha-Beta

                # Start the game if all selections are valid
                if start_button.collidepoint(mouse_pos):
                    if player_color and depth.isdigit() and int(depth) > 0:
                        return player_color, ai_color, int(depth), use_alpha_beta, use_expectimax

            elif event.type == pygame.KEYDOWN and input_active:
                # Handle text input for depth
                if event.key == pygame.K_BACKSPACE:
                    depth = depth[:-1]
                elif event.unicode.isdigit():
                    depth += event.unicode

        clock.tick(30)

    pygame.quit()
    return player_color, ai_color, depth, use_alpha_beta, use_expectimax

def play_game():
    global PLAYER_COLOR, AI_COLOR
    # Setup screen for color, depth, Alpha-Beta pruning, and Expectimax selection
    PLAYER_COLOR, AI_COLOR, depth, use_alpha_beta, use_expectimax = setup_screen()
    print(f"Selected Player Color: {PLAYER_COLOR}")
    print(f"Selected AI Depth: {depth}")
    print(f"Alpha-Beta Pruning: {'Enabled' if use_alpha_beta else 'Disabled'}")
    print(f"Expectiminimax: {'Enabled' if use_expectimax else 'Disabled'}")

    board = create_board()
    turn = 1  # Start with the player

    pygame.init()
    screen = pygame.display.set_mode((COLS * SQUARESIZE, (ROWS + 1) * SQUARESIZE))
    pygame.display.set_caption("Connect Four")
    draw_board(board, screen)

    while not is_full(board):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Player turn
            if turn == 1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x_pos = event.pos[0]
                    col = x_pos // SQUARESIZE
                    if is_valid_move(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, 1)  # Player move
                        draw_board(board, screen)
                        turn = -1  # Switch to AI turn
        # AI turn
        if turn == -1:
            if use_expectimax:
                col, _ = expect_maximize(board, depth)  # Expectimax
            elif use_alpha_beta:
                col = ai_move_with_pruning(board, depth)  # With Alpha-Beta pruning
            else:
                col = ai_move(board, depth)  # Without Alpha-Beta pruning
            
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, -1)  # AI move
            draw_board(board, screen)
            turn = 1  # Switch to player turn
    print(board)
    calculate_final_scores(board)

if __name__ == "__main__":
    play_game()
    print("Game Over!")