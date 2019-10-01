package fr.asenka.detektor.util;

import static java.lang.Math.log;
import static org.apache.commons.math3.stat.StatUtils.sumSq;

import java.util.function.Function;



public class CostFunction implements Function<Params, Double> {

	@Override
	public Double apply(Params p) {

		float lambda = p.getLambda();
		long m = p.getNumberOfExamples();
		long k = p.getNumberOfClasses();
		double[] h = p.getPredictions();
		double[] y = p.getLabels();
		double[] theta = p.getTheta();

		double cost = 0d;

		for (int i = 0; i < m * k; i++)
			cost += y[i] * log(h[i]) + (1d - y[i]) * log(1d - h[i]);

		double regularization = (lambda / (2d * m)) * sumSq(theta);

		return (-1d / m) * cost + regularization;
	}
}
