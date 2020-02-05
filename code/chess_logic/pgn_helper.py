"""
methods for pgn_writer and pgn_reader
"""
#rows are 1-8
#assumed 8x8 2d array
#flipped means display w/ black pieces on bottom
COL_HEADS = {chr(i+97): i for i in range(8)} #ltrs a-h
def display(board, flipped=False):
    if flipped:
        board = board[::-1]

    left_margin = " "*3
    output = "\n"+left_margin
    for k in COL_HEADS:
        output += "  {} ".format(k)
    line = left_margin + '-'*33
    output += "\n" + line
    for r in range(8):
        row_head = " "+str(r+1)+" " if flipped else " "+str(8-r)+" "
        output += "\n{}| ".format(row_head)
        for c in range(7):
            output += str(board[r][c])+' | '
        output += str(board[r][-1])+ ' |{}\n'.format(row_head)
        output += line
    output += "\n"+left_margin
    for k in COL_HEADS:
        output += "  {} ".format(k)
    output += "\n"
    print(output)
    return output

#useful helpers
# def get_col(array, i):
#     return [r[i] for r in array]

def in_bounds(pos): #checking against negative coords or coords > 8
    return pos[0]>=0 and pos[1]>=0 and pos[0]<8 and pos[1]<8

#lookup table
REL_STARTS = {
    "N": [[(1,2),(2,1)],[(-1,2),(-2,1)],[(1,-2),(2,-1)],[(-1,-2),(-2,-1)]],
    "R": [[(0,i) for i in range(-1, -8, -1)], [(0,i) for i in range(1, 8)], [(i,0) for i in range(-1, -8, -1)], [(i,0) for i in range(1, 8)]],
    "B": [[(i,i) for i in range(-1, -8, -1)], [(i,i) for i in range(1, 8)], [(-i,i) for i in range(-1, -8, -1)], [(-i,i) for i in range(1, 8)]],
    "K": [[t] for t in ({(i,j) for i in range(-1, 2) for j in range(-1, 2)} - {(0,0)})],
}
REL_STARTS["Q"] = REL_STARTS["R"]+REL_STARTS["B"] #add in queen moves
