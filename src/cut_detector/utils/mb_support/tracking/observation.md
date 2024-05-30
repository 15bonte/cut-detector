# Regarding the gap_closing/stack issue (fixed):

The file 
20231019-t1_siCep55-50-4_mitosis_21_25_to_117.tiff
was used for the observation

DEFAULT:
[ target_coord ] is a (std) list of the np.array
    target_coord is an np.array
        shape: 4

stack is a 2D npArray (line-col). 
    Number of line is variable (makes sense since this is where we compute gap closing costs)
    Seems to always have n cols, where n is the number of coordinates (makes sense)


SPATIAL right now is partially broken. It produces:
[ target coord ] is OK

stack is fixed now and produces a 2D np array with as many lines as candidates
and m spatial columns
