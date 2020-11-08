import sys
def main():
	training_set=""
	training_labels=""
	test_set=""
	test_labels=""
	model=""    
	coun1=False
	coun2=False
	coun3=False
	coun4=False
	coun5=False
	for i in range(0,10):
		if sys.argv[i]=='-d' and coun1==False:
			coun1=True
			training_set=sys.argv[i+1]
		elif sys.argv[i]=='-d1' and coun2==False:
			coun2=True
			training_labels=sys.argv[i+1]
		elif sys.argv[i]=='-t' and coun3==False:
			coun3=True
			test_set=sys.argv[i+1]
		elif sys.argv[i]=='-t1' and coun4==False:
			coun4=True
			test_labels=sys.argv[i+1]
		elif sys.argv[i]=='-model' and coun5==False:
			coun5=True
			model=sys.argv[i+1]
		if coun1==True and coun2==True and coun3==True and coun4==True and coun5==True:
			break



if __name__ == "__main__":
    main()
