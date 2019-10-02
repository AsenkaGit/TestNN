package fr.asenka.detektor;

import java.util.function.Function;

import fr.asenka.detektor.util.Matrix;

public class NeuralNetwork {

	private static final double EPSILON = 0.0001d;
	
	private static final Function<Double, Double> SIGMOID = z -> 1d / (1d + Math.exp(-z));
//
//	// n 
//	private int numFeatures;
//
//	// k
//	private int numClasses;
//
//	private int numNeuronsHiddenLayer;
//
	private Matrix T1;

	private Matrix T2;

	public NeuralNetwork(int numFeatures, int numClasses, int numNeuronsHiddenLayer) {
//		this.numFeatures = numFeatures;
//		this.numClasses = numClasses;
//		this.numNeuronsHiddenLayer = numNeuronsHiddenLayer;

		this.T1 = Matrix.random(numNeuronsHiddenLayer, numFeatures + 1, -EPSILON, EPSILON);
		this.T2 = Matrix.random(numClasses, numNeuronsHiddenLayer + 1, -EPSILON, EPSILON);
	}

	/**
	 * 
	 * @param inputData
	 * @return
	 */
	public Matrix predict(Matrix inputData) {

		Matrix X = inputData.isColumn() ? inputData : inputData.transpose();
		Matrix zero = Matrix.zeros(1, X.columns());
		
		Matrix A1 = zero.concatV(X);
		Matrix Z2 = T1.multiply(A1);
		Matrix A2 = zero.concatV(Z2.applyOnEach(SIGMOID));
		Matrix Z3 = T2.multiply(A2);
		Matrix A3 = Z3.applyOnEach(SIGMOID);
		
		return A3.indexMaxByColumn();
	}

}
