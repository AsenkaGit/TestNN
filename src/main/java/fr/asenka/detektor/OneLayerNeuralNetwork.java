package fr.asenka.detektor;

import static fr.asenka.detektor.util.Matrix.binaryMatrix;
import static fr.asenka.detektor.util.Matrix.log;
import static fr.asenka.detektor.util.Matrix.sumAll;
import static fr.asenka.detektor.util.Matrix.zeros;

import fr.asenka.detektor.util.Matrix;

public class OneLayerNeuralNetwork extends NeuralNetwork {

	public OneLayerNeuralNetwork(int numFeatures, int numClasses, int numNeuronsHiddenLayer, double learningRate, Matrix data, Matrix labels) {

		this.m = data.rows();
		this.k = numClasses;
		this.n = numFeatures;
		this.h = numNeuronsHiddenLayer;
		this.alpha = learningRate;
		
		this.X = data;
		this.Y = binaryMatrix(labels.transpose(), k).transpose();
		this.T = new Matrix[2];
		this.dT = new Matrix[2];
		this.A = new Matrix[3];
		initializeWeights();
	}
	
	@Override
	public double[] train(int iterations, double correctionAlpha) {
		
		double[] costHistory = new double[iterations + 1];
		
		for(int i = 0; i <= iterations; i++) {
			feedForward(); // Compute A1, A2, A3 and H (H = A3)
			backPropagation(); // Compute gradientTheta1 and gradientTheta2
			gradientDescent(); // Update theta1 and theta2
			costHistory[i] = computeCost(); // Compute J
			System.out.println("[" + i + "] alpha = " + alpha + "\tcost = " + costHistory[i]);
			this.alpha += correctionAlpha;
		}
		return costHistory;
	}

	private void initializeWeights() {
		
		this.T[0] = Matrix.random(h, n + 1, -0.5, 0.5);
		this.T[1] = Matrix.random(k, h + 1, -0.5, 0.5);
	}

	private void feedForward() {

		Matrix Z2, Z3;
		Matrix ones = Matrix.ones(X.rows(), 1);

		A[0] = ones.concatH(X);
		Z2 = A[0].multiply(T[0].transpose());
		A[1] = ones.concatH(Z2.applyOnEach(SIGMOID));
		Z3 = A[1].multiply(T[1].transpose());
		A[2] = Z3.applyOnEach(SIGMOID); 
		H = A[2].copy();
	}
	
	private double computeCost() {
		return cost(H, Y, m) + regularization(m, T[0], T[1]);
	}
	
	private void backPropagation() {
		
		Matrix a1, a2, a3, y, d2, d3, D1, D2, regTheta1, regTheta2;
		Matrix delta1 = Matrix.zeros(T[0].rows() + 1, T[0].columns());
		Matrix delta2 = Matrix.zeros(T[1].rows(), T[1].columns());
		
		for (int i = 0; i < m; i++) {
			a3 = A[2].getRow(i).transpose();
			a2 = A[1].getRow(i).transpose();
			a1 = A[0].getRow(i).transpose();
			y = Y.getRow(i).transpose();
			
			d3 = a3.subtract(y);
			d2 = T[1].transpose().multiply(d3).multiplyEachEntry(a2.multiplyEachEntry(a2.negative().add(1)));
			
			delta2 = delta2.add(d3.multiply(a2.transpose()));
			delta1 = delta1.add(d2.multiply(a1.transpose()));
		}
		D1 = delta1.subMatrix(1, 0).divide(m);
		D2 = delta2.divide(m);
		
		regTheta1 = T[0].copy();
		regTheta2 = T[1].copy();
		regTheta1.setColumn(0, zeros(regTheta1.rows(), 1));
		regTheta2.setColumn(0, zeros(regTheta2.rows(), 1));
		
		// We get all the partial derivatives of the weights in theta1 and theta2
		dT[0] = D1.add(regTheta1.multiply(LAMBDA / m));
		dT[1] = D2.add(regTheta2.multiply(LAMBDA / m));
	}
	
	private void gradientDescent() {
		updateWeigths(T[0], dT[0]);
		updateWeigths(T[1], dT[1]);
	}
	
	private void updateWeigths(Matrix weights, Matrix derivWeights) {
		
		for (int r = 0; r < weights.rows(); r++) 
			for (int c = 0; c < weights.columns(); c++) {
					double theta = weights.get(r, c);
					double derivTheta = derivWeights.get(r, c);
					double regularization = (LAMBDA / m) * theta;
					weights.set(r, c,  theta - alpha * derivTheta  + regularization) ;
			}
	}

	public static final Matrix predict(Matrix X, Matrix[] weights) {
		
		Matrix Z2, Z3, A1, A2, A3, T1, T2;
		T1 = weights[0];
		T2 = weights[1];
		Matrix ones = Matrix.ones(X.rows(), 1);
	
		A1 = ones.concatH(X);
		Z2 = A1.multiply(T1.transpose());
		A2 = ones.concatH(Z2.applyOnEach(SIGMOID));
		Z3 = A2.multiply(T2.transpose());
		A3 = Z3.applyOnEach(SIGMOID); 
		
		return A3.indexMaxByRow();
	}

	private static final double cost(Matrix predictions, Matrix labels, int numExamples) {
		
		double m = (double) numExamples;
		Matrix Y = labels;
		Matrix H = predictions;
		Matrix ones = Matrix.ones(H.rows(), H.columns());
		
		Matrix M1 = (Y.negative()).multiplyEachEntry(log(H));
		Matrix M2 = (ones.subtract(Y)).multiplyEachEntry(log(ones.subtract(H)));
		
		return sumAll(M1.subtract(M2)) / m;
	}

	private static final double regularization(int numExamples, Matrix theta1, Matrix theta2) {
		
		double m = (double) numExamples;
		double t1 = sumAll(theta1.subMatrix(0, 1).applyOnEach(x -> x * x));
		double t2 = sumAll(theta2.subMatrix(0, 1).applyOnEach(x -> x * x));
		return (LAMBDA / (2d * m)) * (t1 + t2);
	}
}
