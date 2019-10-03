package fr.asenka.detektor;

import static fr.asenka.detektor.util.Matrix.binaryMatrix;
import static fr.asenka.detektor.util.Matrix.log;
import static fr.asenka.detektor.util.Matrix.sum;
import static fr.asenka.detektor.util.Matrix.sumAll;
import static fr.asenka.detektor.util.Matrix.zeros;

import java.util.function.Function;

import fr.asenka.detektor.util.Matrix;

public class NeuralNetwork {

	private static final double LAMBDA = 2d;
	
	private static final double ALPHA = 1d;

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
	}
	
	public Matrix predict(Matrix data) {
		
		Matrix Z2, Z3;
		Matrix ones = Matrix.ones(X.rows(), 1);

		A1 = ones.concatH(X);
		Z2 = A1.multiply(theta1.transpose());
		A2 = ones.concatH(Z2.applyOnEach(SIGMOID));
		Z3 = A2.multiply(theta2.transpose());
		A3 = Z3.applyOnEach(SIGMOID); 
		
		return A3.indexMaxByRow();
	}

	public void train() {
		
		double costRegularized;
		
		initializeWeights();
		feedForward();
		costRegularized = cost(H, Y, m) + regularization(m, theta1, theta2);
		System.out.println("cost = " + costRegularized);
		
		
		for (int i = 0; i < 10; i++) {
			
			backPropagation();
			gradientDescent();
			feedForward();
			costRegularized = cost(H, Y, m) + regularization(m, theta1, theta2);
			System.out.println("cost = " + costRegularized);
		}
	}
	
	private void initializeWeights() {
		double epsilon1 = Math.sqrt(6) / Math.sqrt(h + n + 1);
		double epsilon2 = Math.sqrt(6) / Math.sqrt(k + h + 1);
		
		this.theta1 = Matrix.random(h, n + 1, -epsilon1, epsilon1);
		this.theta2 = Matrix.random(k, h + 1, -epsilon2, epsilon2);
	}

	private void feedForward() {

		Matrix Z2, Z3;
		Matrix ones = Matrix.ones(X.rows(), 1);

		A1 = ones.concatH(X);
		Z2 = A1.multiply(theta1.transpose());
		A2 = ones.concatH(Z2.applyOnEach(SIGMOID));
		Z3 = A2.multiply(theta2.transpose());
		A3 = Z3.applyOnEach(SIGMOID); 
		H = A3;
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
		
		theta1 = updateTheta(theta1, gradientTheta1, m);
		theta2 = updateTheta(theta2, gradientTheta2, m);
	}
	
	private static final Matrix updateTheta(Matrix theta, Matrix gradientTheta, int m) {
		
		int rows = theta.rows();
		int columns = theta.columns();
		
		Matrix result = new Matrix(theta.rows(), theta.columns());
		
		for (int r = 0; r < rows; r++)
			for (int c = 0; c < columns; c++) 
				result.set(r, c, theta.get(r, c) - ALPHA * gradientTheta.get(r, c) + ((LAMBDA / m) * theta.get(r, c) )); 

		return result;
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
