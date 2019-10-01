package fr.asenka.detektor.util;

import org.apache.commons.math3.linear.RealMatrix;

public class Params {
	
	private long m;
	
	private long k;
	
	private float lambda;
	
	private double[] h;
	
	private double[] y;
	
	private double[] theta;

	private Params(long m, long k, float lambda, double[] h, double[] y, double[] theta) {
		this.m = m;
		this.k = k;
		this.lambda = lambda;
		this.h = h;
		this.y = y;
		this.theta = theta;
	}
	
	public static final Params with(long m, long k, float lambda, RealMatrix H, RealMatrix Y, RealMatrix... thetas) {
		
		double[] h = CustomMatrixUtils.vectorizeMatrix(H);
		double[] y = CustomMatrixUtils.vectorizeMatrix(Y);
		
		if (h.length != y.length)
			throw new IllegalArgumentException("h and y do not have the same size");
		
		double[] theta = CustomMatrixUtils.vectorizeMatrices(thetas);
		return new Params(m, k, lambda, h, y, theta);
	}

	public long getNumberOfExamples() {
		return m;
	}

	public long getNumberOfClasses() {
		return k;
	}

	public float getLambda() {
		return lambda;
	}

	public double[] getPredictions() {
		return h;
	}

	public double[] getLabels() {
		return y;
	}

	public double[] getTheta() {
		return theta;
	}
}
