package fr.asenka.detektor;

import fr.asenka.detektor.util.Matrix;

public class App {

	public static void main(String[] args) {

		DataSet trainingData = new DataSet("src/main/resources/train-images-idx3-ubyte.gz",
				"src/main/resources/train-labels-idx1-ubyte.gz");

		Matrix X = trainingData.getVectorizedImages().normalize();
		Matrix y = trainingData.getVectorizedLabels();

		NeuralNetwork nn = new NeuralNetwork(28 * 28, 10, 25, X, y);
		nn.train();
		Matrix p = nn.predict(X);
		
		System.out.println(y.rows(10, 20).transpose());
		System.out.println(p.rows(10, 20).transpose());
	}
}
