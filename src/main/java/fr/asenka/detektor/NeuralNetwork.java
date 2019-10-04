package fr.asenka.detektor;

import static fr.asenka.detektor.util.Matrix.binaryMatrix;
import static fr.asenka.detektor.util.Matrix.log;
import static fr.asenka.detektor.util.Matrix.sum;
import static fr.asenka.detektor.util.Matrix.sumAll;
import static fr.asenka.detektor.util.Matrix.zeros;

import java.util.function.Function;

import fr.asenka.detektor.util.Matrix;

public class NeuralNetwork {
	
	private static final double LAMBDA = 1d;
	
	private static final double ALPHA = 2.5d;

	private static final Function<Double, Double> SIGMOID = z -> 1d / (1d + Math.exp(-z));

	private int k; // classes 
	
	private int m; // examples 
	
	private int n; // features 
	
	private int h; // hidden layer size
	
	private Matrix X; // data

	private Matrix Y; // binarized labels

	private Matrix theta1; // weights from input layer to hidden layer
 
	private Matrix theta2; // weights from hidden layer to output layer
	
	private Matrix gradientTheta1; // partial derivatives of theta1
	
	private Matrix gradientTheta2; // partial derivatives of theta2
	
	private Matrix A1;
	
	private Matrix A2;
	
	private Matrix A3;
	
	private Matrix H;

	public NeuralNetwork(int numFeatures, int numClasses, int numNeuronsHiddenLayer, Matrix data, Matrix labels) {

		this.m = data.rows();
		this.k = numClasses;
		this.n = numFeatures;
		this.h = numNeuronsHiddenLayer;
		
		this.X = data;
		this.Y = binaryMatrix(labels.transpose(), k).transpose();
		initializeWeights();
	}
	
	public double[] train(int iterations) {

		double[] costHistory = new double[iterations + 1];
		feedForward();
		costHistory[0] = cost(H, Y, m) + regularization(m, theta1, theta2);
		
		for(int i = 1; i <= iterations; i++) {
			backPropagation();
			gradientDescent();
			feedForward();
			costHistory[i] = cost(H, Y, m) + regularization(m, theta1, theta2);
			System.out.println("[" + i + "] cost = " + costHistory[i]);
		}
		return costHistory;
	}

	public Matrix getTheta1() {
		return theta1;
	}

	public Matrix getTheta2() {
		return theta2;
	}

	private void initializeWeights() {
		this.theta1 = Matrix.random(h, n + 1, -0.5, 0.5);
		this.theta2 = Matrix.random(k, h + 1, -0.5, 0.5);
	}

	private void feedForward() {

		Matrix Z2, Z3;
		Matrix ones = Matrix.ones(X.rows(), 1);

		A1 = ones.concatH(X);
		Z2 = A1.multiply(theta1.transpose());
		A2 = ones.concatH(Z2.applyOnEach(SIGMOID));
		Z3 = A2.multiply(theta2.transpose());
		A3 = Z3.applyOnEach(SIGMOID); 
		H = A3.copy();
	}
	
	private void backPropagation() {
		
		Matrix a1, a2, a3, y, d2, d3, D1, D2, regTheta1, regTheta2;
		Matrix delta1 = Matrix.zeros(theta1.rows() + 1, theta1.columns());
		Matrix delta2 = Matrix.zeros(theta2.rows(), theta2.columns());
		
		for (int i = 0; i < m; i++) {
			a3 = A3.getRow(i).transpose();
			a2 = A2.getRow(i).transpose();
			a1 = A1.getRow(i).transpose();
			y = Y.getRow(i).transpose();
			
			d3 = a3.subtract(y);
			d2 = theta2.transpose().multiply(d3).multiplyEachEntry(a2.multiplyEachEntry(a2.negative().add(1)));
			
			delta2 = delta2.add(d3.multiply(a2.transpose()));
			delta1 = delta1.add(d2.multiply(a1.transpose()));
		}
		
		D1 = delta1.subMatrix(1, 0).divide(m);
		D2 = delta2.divide(m);
		
		regTheta1 = theta1.copy();
		regTheta2 = theta2.copy();
		regTheta1.setColumn(0, zeros(regTheta1.rows(), 1));
		regTheta2.setColumn(0, zeros(regTheta2.rows(), 1));
		
		gradientTheta1 = D1.add(regTheta1.multiply(LAMBDA / m));
		gradientTheta2 = D2.add(regTheta2.multiply(LAMBDA / m));
	}
	
	private void gradientDescent() {
		updateWeigths(theta1, gradientTheta1);
		updateWeigths(theta2, gradientTheta2);
	}
	
	private void updateWeigths(Matrix weights, Matrix derivWeights) {
		
		int rows = weights.rows();
		int columns = weights.columns();
		
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < columns; c++) {
					double theta = weights.get(r, c);
					double derivTheta = derivWeights.get(r, c);
					weights.set(r, c,  theta - ALPHA * derivTheta  + (LAMBDA / m) * theta) ;
			}
		}
	}

	public static final Matrix predict(Matrix X, Matrix theta1, Matrix theta2) {
		
		Matrix Z2, Z3, A1, A2, A3;
		Matrix ones = Matrix.ones(X.rows(), 1);
	
		A1 = ones.concatH(X);
		Z2 = A1.multiply(theta1.transpose());
		A2 = ones.concatH(Z2.applyOnEach(SIGMOID));
		Z3 = A2.multiply(theta2.transpose());
		A3 = Z3.applyOnEach(SIGMOID); 
		
		return A3.indexMaxByRow();
	}
	
	public static final int countCorrectPredictions(Matrix predictions, Matrix labels) {
		
		int size = predictions.rows();
		int correctPredictions = 0;
		
		for (int i = 0; i < size; i++)
			correctPredictions += labels.get(i, 0) != predictions.get(i, 0) ? 0 : 1;
		
		return correctPredictions;
	}

	private static final double regularization(int numExamples, Matrix theta1, Matrix theta2) {
		
		double m = (double) numExamples;
		double t1 = sumAll(theta1.subMatrix(0, 1).applyOnEach(x -> x * x));
		double t2 = sumAll(theta2.subMatrix(0, 1).applyOnEach(x -> x * x));
		return (LAMBDA / (2d * m)) * (t1 + t2);
	}
	
	private static final double cost(Matrix predictions, Matrix labels, int numExamples) {
		
		double m = (double) numExamples;
		Matrix Y = labels;
		Matrix H = predictions;
		Matrix ones = Matrix.ones(H.rows(), H.columns());
		
		Matrix M1 = (Y.negative()).multiplyEachEntry(log(H));
		Matrix M2 = (ones.subtract(Y)).multiplyEachEntry(log(ones.subtract(H)));
		
		return sum(sum(M1.subtract(M2))).get(0, 0) / m;
	}
}
