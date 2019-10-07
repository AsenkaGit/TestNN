package fr.asenka.detektor;

import java.util.function.Function;

import fr.asenka.detektor.util.Matrix;

public abstract class NeuralNetwork {

	public static final double DEFAULT_ALPHA = 2.5d;

	protected static final double LAMBDA = 1d;

	protected double alpha = 2.5d;

	protected static final Function<Double, Double> SIGMOID = z -> 1d / (1d + Math.exp(-z));

	protected int k; // classes

	protected int m; // examples

	protected int n; // features

	protected int h; // hidden layer size

	protected Matrix X; // data

	protected Matrix Y; // binarized labels

	protected Matrix[] T;

	protected Matrix[] dT;

	protected Matrix A[];

	protected Matrix H;

	public double[] train(int iterations) {
		return train(iterations, 0d);
	}

	public abstract double[] train(int iterations, double alphaCorrection);

	public Matrix[] getWeights() {
		return T;
	}

	public static final int countCorrectPredictions(Matrix predictions, Matrix labels) {

		int size = predictions.rows();
		int correctPredictions = 0;

		for (int i = 0; i < size; i++)
			correctPredictions += labels.get(i, 0) != predictions.get(i, 0) ? 0 : 1;

		return correctPredictions;
	}
}
