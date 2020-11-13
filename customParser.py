import argparse


#filepath=sys.argv[1]

ap = argparse.ArgumentParser()

ap.add_argument("--seed",
   help="first operand", default='10')

ap.add_argument("--agent",
   help="second operand", default='')

ap.add_argument("--kingsMove",
   help="third operand", default='0')

ap.add_argument("--stochasticity",
   help="fourth operand", default='0')

ap.add_argument("--timeSteps",
   help="fifth operand", default='8000')

ap.add_argument("--epsilon",
   help="sixth operand", default='0.1')
ap.add_argument("--alpha",
   help="seventh operand", default='0.5')

ap.add_argument("--fileName",
   help="eighth operand", default='')
args = vars(ap.parse_args())

def getArg(name):
   return args[name]

def printOutput(regret):
   print("%s, %s, %s, %s, %s, %s"%(args['instance'], args['algorithm'], args['randomSeed'], args['epsilon'], args['horizon'],regret))


