echo "Task1 started..."
python3 executeGridWorld.py --agent sarsa --fileName Task1.png
echo "Task1 Completed."
echo ""

echo "Task2 started..."
python3 executeGridWorld.py --agent sarsa --kingsMove 1 --fileName Task2.png
echo "Task2 Completed."
echo ""

echo "Task3 started..."
python3 executeGridWorld.py --agent sarsa --kingsMove 1 --stochasticity 1 --fileName Task3.png
echo "Task3 Completed."
echo ""

echo "Task4 started..."
python3 executeGridWorld.py --fileName Task4.png
echo "Task4 Completed."
echo ""
