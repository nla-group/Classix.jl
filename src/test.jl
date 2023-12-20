push!(LOAD_PATH, pwd())
using Classix
using MAT
using TickTock

file = matopen("Phoneme_z.mat")
data = read(file, "data") 
close(file)



tic = time()
labels, explain, out = classix(data, radius=0.445, minPts=8, merge_tiny_groups=true)
print("\ntotal classix.jl runtime: ", time()-tic)

#print(out)
#print(labels)