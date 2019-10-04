package fr.asenka.detektor;

import java.io.IOException;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

import fr.asenka.detektor.util.MatLabDataSet;
import fr.asenka.detektor.util.Matrix;

public class App {

	public static void main(String[] args) throws IOException {

		MatLabDataSet ds = new MatLabDataSet();
		Matrix X = ds.getImages();
		Matrix y = ds.getLabels();

//		System.out.println("Training model...");
//		NeuralNetwork nn = new NeuralNetwork(20 * 20, 10, 30, X, y);
//		double[] history = nn.train(200);
//		
		Plot2DPanel plot = new Plot2DPanel();
//		plot.addLinePlot("Cost", history);
		plot.addBoxPlot("Image", Matrix.reshape(X.getRow(0), 20, 20).getRawData());
		
		
		JFrame frame = new JFrame("Final X-Y Data");
        frame.setContentPane(plot);
        frame.setSize(600, 600);
        frame.setVisible(true);
		
//		Matrix p = NeuralNetwork.predict(X, nn.getTheta1(), nn.getTheta2());
//		
//		System.out.println("Correct predictions : " + NeuralNetwork.countCorrectPredictions(p, y) + "/" + p.rows());
		
//		MatLabDataSet.saveWeights(nn.getTheta1(), nn.getTheta2());
	}
}
