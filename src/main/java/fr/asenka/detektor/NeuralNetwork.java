package fr.asenka.detektor;

import static org.apache.commons.math3.linear.MatrixUtils.*;
import static fr.asenka.detektor.util.CustomMatrixUtils.*;

import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class NeuralNetwork {

	private static final double EPSILON = 0.0001d;

	// n 
	private int numFeatures;

	// k
	private int numClasses;

	private int numNeuronsHiddenLayer;

	private RealMatrix T1;

	private RealMatrix T2;
	
//	private RealMatrix gradientT1;
//	
//	private RealMatrix gradientT2;

	public NeuralNetwork(int numFeatures, int numClasses, int numNeuronsHiddenLayer) {
		this.numFeatures = numFeatures;
		this.numClasses = numClasses;
		this.numNeuronsHiddenLayer = numNeuronsHiddenLayer;

		this.T1 = randomMatrix(numNeuronsHiddenLayer, numFeatures + 1, EPSILON); // 
		this.T2 = randomMatrix(numClasses, numNeuronsHiddenLayer + 1, EPSILON);
	}

	/**
	 * 
	 * @param inputData
	 * @return
	 */
	public RealMatrix predict(RealMatrix inputData) {

		RealMatrix X = inputData.getColumnDimension() == 1 ? inputData : inputData.transpose();
		RealMatrix A1 = addSlope(X);
		RealMatrix Z2 = T1.multiply(A1);
		RealMatrix A2 = addSlope(sigmoid(Z2));
		RealMatrix Z3 = T2.multiply(A2);
		RealMatrix A3 = sigmoid(Z3); // H = A3
		
		return indexMax(A3);
	}
	
	/**
	 * 
	 * @param set
	 */
	public void train(DataSet set) {
		
		
		
	}

	private void backPropagation(long m, RealMatrix X, RealMatrix Y) {
		
		RealMatrix a1, a2, a3, y, d2, d3, D1, D2, reg1, reg2;
		
		RealMatrix A1 = addSlope(X);
		RealMatrix A2 = addSlope(sigmoid(T1.multiply(A1)));
		RealMatrix A3 = sigmoid(T2.multiply(A2)); 
		
		RealMatrix t2Transposed = T2.transpose();
		RealMatrix delta1 = createZeroMatrix(numNeuronsHiddenLayer + 1, numFeatures + 1);
		RealMatrix delta2 = createZeroMatrix(numClasses, numNeuronsHiddenLayer + 1);
		
		for (int i = 0; i < m; i++) {
			
			a1 = createColumnRealMatrix(A1.getColumn(i));
			a2 = createColumnRealMatrix(A2.getColumn(i));
			a3 = createColumnRealMatrix(A3.getColumn(i));
			y = createColumnRealMatrix(Y.getColumn(i));

			// d3 = a3 - y
			d3 = a3.subtract(y); 
			
			// d2 = (Theta2' * d3 )  .* ( a2 .* (1 - a2) );
			d2 = scalarMultiply(t2Transposed.multiply(d3), scalarMultiply(a2, a2.scalarAdd(-1)));
			
			// Delta2 = Delta2 + (d3 * a2');
			// Delta1 = Delta1 + (d2 * a1');
			delta2 = delta2.add(d3.multiply(a2.transpose()));
			delta1 = delta1.add(d2.multiply(a1.transpose()));
		}
		
		D1 = getSubMatrix(delta1, 1, 0).scalarMultiply(1.0 / (double) m);
		D2 = delta2.scalarMultiply(1.0 / (double) m);
		
		reg1 = T1.copy();
		reg2 = T2.copy();
	}

	/**
	 * 
	 * @param m
	 * @return
	 */
	private static final RealMatrix addSlope(RealMatrix m) {

		int rows = m.getRowDimension();
		int columns = m.getColumnDimension();

		RealMatrix result = createRealMatrix(rows + 1, columns);

		for (int row = -1; row < rows; row++)
			for (int column = 0; column < columns; column++)
				result.setEntry(row + 1, column, row < 0 ? 1d : m.getEntry(row, column));

		return result;
	}

	/**
	 * 
	 * @param m
	 * @return
	 */
	private static final RealMatrix indexMax(RealMatrix m) {
		
		int columns = m.getColumnDimension();
		
		// Create a row matrix
		RealMatrix result = MatrixUtils.createRealMatrix(1, columns);
		
		for (int column = 0; column < columns; column++)
			result.setEntry(0, column, indexMax(m.getColumn(column)));
		
		return result;
	}

	/**
	 * 
	 * @param array
	 * @return
	 */
	private static final int indexMax(double[] array) {
		double largest = array[0];
		int index = 0;
		
		for (int i = 1; i < array.length; i++)
			if (array[i] >= largest) {
				largest = array[i];
				index = i;
			}
		return index;
	}

	/**
	 * 
	 * @param rows
	 * @param columns
	 * @param epsilon
	 * @return
	 */
	private static final RealMatrix randomMatrix(int rows, int columns, double epsilon) {

		RealMatrix random = MatrixUtils.createRealMatrix(rows, columns);
		Random rng = new Random();

		final double max = epsilon;
		final double min = -epsilon;

		for (int row = 0; row < rows; row++)
			for (int col = 0; col < columns; col++)
				random.setEntry(row, col, min + (max - min) * rng.nextDouble());

		return random;
	}

	/**
	 * 
	 * @param m
	 * @return
	 */
	private static final RealMatrix sigmoid(RealMatrix m) {

		int rows = m.getRowDimension();
		int columns = m.getColumnDimension();

		RealMatrix result = MatrixUtils.createRealMatrix(rows, columns);

		for (int row = 0; row < rows; row++)
			for (int col = 0; col < columns; col++)
				result.setEntry(row, col, sigmoid(m.getEntry(row, col)));

		return result;
	}

	/**
	 * 
	 * @param z
	 * @return
	 */
	private static final double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}
}
