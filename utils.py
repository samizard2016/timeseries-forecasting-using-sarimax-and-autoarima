import itertools
class Utils:
    @staticmethod
    def get_exog(dep,exog_fixed,exog_others):
        for i in range(len(exog_others)+1):
            if i==0:
                comb = list(itertools.combinations(exog_others,i))
            else:
                comb = comb + list(itertools.combinations(exog_others,i))
        for i in range(len(comb)):
            comb[i] = tuple([dep]) + tuple(exog_fixed) + comb[i]
        return comb

if __name__=='__main__':
    x = ['a','b','c','d','e','f']
    dep = x[0]
    fixed = x[1]
    others = x[2:]
    print(f"dep {dep}\n fixed {fixed}\n others {others}\n")
    print(Utils.get_exog(dep,fixed,others))
