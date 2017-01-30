f=open("wine.txt")
f1=open("wine.csv","w")
for line in f:
        if(line.startswith("\"")):
            continue
        l=line.replace("\n","").split(";")
        val=int(l[-1])
        good=-1
        if(val>=7):
            good=1
        str1=""
        for i in range(0,len(l)-1):
        	str1=str1+str(l[i])+","
        str1=str1+str(good)
        f1.write(str1+"\n")


