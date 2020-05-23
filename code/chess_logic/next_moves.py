import sys
import pgn_reader as reader
import pgn_helper as ph

"""
given one piece and current board,
figure out all next poss positions for that piece
doesn't handle checks

st_pos in (r,c)
board must be oriented std (white-) or black-on-bottom
"""
def next_poss_from_st(board, st_pos):
    next_poss = set()
    piece = board[st_pos[0]][st_pos[1]]
    if piece == "-": return set()

    upper_piece = piece.upper()
    if upper_piece == "P":
        fwd = 1 #black pawn assumed
        if piece.isupper(): #white pawn
            fwd = -1

        #move fwd 2 if unmoved
        if st_pos[0] in {1, 6}:
            poss = [(st_pos[0]+fwd, st_pos[1]), (st_pos[0]+(2*fwd), st_pos[1])]
            for p in poss:
                if not ph.in_bounds(p): continue
                if board[p[0]][p[1]] != "-": break
                next_poss.add(p)
        #en passant
        #actually not possible to figure out en passant w/out prev board state
        #but assuming en passant is possible if the current board appears to allow it
        elif (st_pos[0] == 3 and piece.isupper()) or (st_pos[0] == 4 and piece.islower()):
            for shift in {-1, 1}:
                p = (st_pos[0], st_pos[1]+shift)
                if not ph.in_bounds(p): continue
                opp_team = (board[p[0]][p[1]].isupper() and piece.islower()) or (piece.isupper() and board[p[0]][p[1]].islower())
                if board[p[0]][p[1]].upper() == "P" and opp_team:
                    if board[p[0]+(2*fwd)][p[1]] == "-":
                        next_poss.add((p[0]+fwd, p[1]))

        #move fwd 1 otherwise (could overlap w/ above, but set fixes that)
        p = (st_pos[0]+fwd, st_pos[1])
        if ph.in_bounds(p):
            if board[p[0]][p[1]] == "-":
                next_poss.add(p)

        #pawn captures
        for shift in {-1, 1}:
            p = (st_pos[0]+fwd, st_pos[1]+shift)
            if ph.in_bounds(p):
                if board[p[0]][p[1]] != "-":
                    opp_team = (board[p[0]][p[1]].isupper() and piece.islower()) or (piece.isupper() and board[p[0]][p[1]].islower())
                    if opp_team:
                        next_poss.add(p)
    else:
        #all pieces besides pawns
        starts = ph.REL_STARTS[upper_piece]
        search_sp = [[(st_pos[0]+p[0], st_pos[1]+p[1]) for p in dir] for dir in starts]

        st_poss = set()
        for dir in search_sp:
            for pos in dir:
                if ph.in_bounds(pos):
                    #capture handler
                    if board[pos[0]][pos[1]] != "-":
                        pc = board[pos[0]][pos[1]]
                        opp_team = (pc.isupper() and piece.islower()) or (piece.isupper() and pc.islower())
                        # print(opp_team, pos)
                        if opp_team:
                            next_poss.add(pos)
                        if upper_piece != "N": #not sure if in {"N", "n"} is faster
                            break
                    else:
                        next_poss.add(pos)

        #castling check
        if upper_piece in {"K", "R"}:
            #always in king, rook order
            pairs_to_check = set()
            if st_pos[0] in {0, 7}:
                if board[st_pos[0]][st_pos[1]].upper() == "K" and st_pos[1] == 4:
                    for col in [0, 7]:
                        if board[st_pos[0]][col].upper() == "R":
                            pairs_to_check.add((st_pos, (st_pos[0], col)))
                elif board[st_pos[0]][st_pos[1]].upper() == "R" and st_pos[1] in {0, 7}:
                    if board[st_pos[0]][4].upper() == "K":
                        pairs_to_check.add(((st_pos[0], 4), st_pos))

            for p1, p2 in pairs_to_check:
                low = min(p1[1], p2[1])
                high = max(p1[1], p2[1])

                good_castle = True
                for c in range(low+1, high):
                    if board[p1[0]][c] != "-":
                        good_castle = False
                        break

                if good_castle:
                    #map of every possible pair to where pair will end up
                    castle_map = {
                        ((0,4),(0,0)):((0,2),(0,3)),
                        ((0,4),(0,7)):((0,6),(0,5)),
                        ((7,4),(7,0)):((7,2),(7,3)),
                        ((7,4),(7,7)):((7,6),(7,5)),
                    }
                    if upper_piece == "K":
                        next_poss.add(castle_map[(p1, p2)][0])
                    else:
                        next_poss.add(castle_map[(p1, p2)][1])

    return next_poss

"""
turn poss set into bitboard
"""
def get_bitboard_from_poss(next_poss):
    bitboard = [[0 for c in range(8)] for r in range(8)]
    for r, c in next_poss:
        bitboard[r][c] = 1
    return bitboard

"""
take board state, get all possible next board states
return as a 3D array
where every sqr has set of every piece that could be there
(includes possibility of pieces not moving)
"""
def get_stacked_poss(board):
    piece_bitboards = {}
    stacked_poss = [[{board[r][c]} for c in range(8)] for r in range(8)]

    for r in range(8):
        for c in range(8):
            #liberally apply en passant possibility
            if (r == 3 and piece == "P") or (r == 4 and piece == "p"):
                stacked_poss[r][c].add("-")

            piece = board[r][c]
            next_poss = next_poss_from_st(board, (r, c))
            if next_poss: #piece can move, meaning current sqr could be vacated
                stacked_poss[r][c].add("-")

            if piece not in piece_bitboards:
                piece_bitboards[piece] = get_bitboard_from_poss(next_poss)
            else: #merge bitboards of same piece type
                to_add = get_bitboard_from_poss(next_poss)
                for tr in range(8):
                    for tc in range(8):
                        if to_add[tr][tc]:
                            piece_bitboards[piece][tr][tc] = 1

    for piece, bitboard in piece_bitboards.items():
        for r in range(8):
            for c in range(8):
                if bitboard[r][c]:
                    stacked_poss[r][c].add(piece)

                    if (piece=="P" and r==0) or (piece=="p" and r==7):
                        # print(piece, "promotable at", (r,c))
                        # for row in bitboard:
                        #     print(row)

                        for promotable in {"Q","R","N","B"}:
                            if piece.isupper():
                                stacked_poss[r][c].add(promotable)
                            else:
                                stacked_poss[r][c].add(promotable.lower())

    return stacked_poss

#check whether given next board is possible given current board's stacked_poss
def is_next_board_poss(board, stacked_poss):
    for r in range(8):
        for c in range(8):
            if board[r][c] not in stacked_poss[r][c]:
                print("not poss here")
                print(r, c)
                print(board[r][c])
                print(stacked_poss[r][c])
                return False
    return True

"""
give a game in pgn file format, will play through and see if any illegal moves made
(according to stacked_poss, so checks not accounted for)
"""
def main():
    pgn_file = sys.argv[1]
    num_move_list = reader.clean_pgn(pgn_file)
    for i in range(len(num_move_list)):
        print(i, end=". ")
        print(num_move_list[i])
    move_list = reader.flatten_move_list(num_move_list)

    board = reader.FEN_to_board()
    ph.display(board)

    for i in range(len(move_list)):
        stacked_poss = get_stacked_poss(board)

        next_board = reader.make_move(board, move_list[i], (i+1)%2)

        # print(move_list[i])
        # ph.display(next_board)
        # for row in stacked_poss:
        #     print(row)

        if not is_next_board_poss(next_board, stacked_poss):
            ph.display(board)
            print("IMPOSSIBLE", i, move_list[i])
            ph.display(next_board)

            print("stacked_poss:")
            for row in stacked_poss:
                print(row)

            break
        else:
            print("move {} good".format(i))

        board = next_board

if __name__ == '__main__':
    main()
