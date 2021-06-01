def sub_cost(cell, c1, c2, cost):
    if c1 != c2:
        return cell + cost
    else :
        return cell

#edit_distance(word, otherword, len(word), len(otherword), norm, word_conf, otherword_conf)
def edit_distance(str1, str2, m, n, str1_conf, str2_conf):
    """Fill the str1 x str2 matrix, and get the the edit distance normalized by string length"""
    norm1 = sum(str1_conf)#max(str1_conf) - min(str1_conf)
    norm2 = sum(str2_conf)#max(str2_conf) - min(str2_conf)
    # Create a table to store results of subproblems
    d = [[0 for x in range(n)] for x in range(m)]
    for i in range(m):
        for j in range(n):
            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                # Insertions entail inserting the char in str2.
                # We therefore use the weight for the char in str2.
                d[i][j] = str2_conf[j]
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                # Deletions entail deleting the char in str1.
                # We therefore use the weight for the char in str1.
                d[i][j] = str1_conf[i]
            else:
                # Local costs of each operation, where we use LM confidence in the
                # character being inserted or deleted, or the average of the 2 for substitution
                d_conf = (str2_conf[j]/norm2)
                i_conf = (str1_conf[i]/norm1)
                s_conf = ((d_conf + i_conf) / 2)
                # Dynamic cost of each operation + the path to it.
                # We take 1 minus the conf so that higher confidence (i.e. hypothesized affixes)
                # Corresponds to lower distance
                delete = d[i-1][j] + (1 - d_conf)
                insert = d[i][j-1] + (1 - i_conf)
                sub = sub_cost(
                    cell=d[i-1][j-1],
                    c1=str1[i-1],
                    c2=str2[j-1],
                    cost=(1 - s_conf)
                )
                d[i][j] = min(delete, insert, sub)

    return d[m-1][n-1]/max(len(str1),len(str2))
