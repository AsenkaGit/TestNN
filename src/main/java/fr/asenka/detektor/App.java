package fr.asenka.detektor;

import java.io.IOException;

import javax.swing.JFrame;

import org.math.plot.Plot2DPanel;

import fr.asenka.detektor.util.DataSet;
import fr.asenka.detektor.util.Matrix;
import fr.asenka.detektor.util.MnistDataSet;

public class App {

	public static void main(String[] args) throws IOException {

		DataSet ds = new MnistDataSet();
		Matrix X = ds.getImages().rows(0, 2999);
		Matrix y = ds.getLabels().rows(0, 2999);
		
		System.out.println("Training model with alpha=2.5...");
		OneLayerNeuralNetwork nn1 = new OneLayerNeuralNetwork(ds.getImageSize(), 10, 20, 2.5d, X, y);
		
		plotCostHistory(nn1.train(50));
        
        Matrix testX = ds.getImages().rows(40000, 59999);
        Matrix testy = ds.getLabels().rows(40000, 59999);
		Matrix p = OneLayerNeuralNetwork.predict(testX, nn1.getWeights());
		System.out.println("Correct predictions : " + OneLayerNeuralNetwork.countCorrectPredictions(p, testy) + "/" + p.rows());
		
	}
	
	private static final void plotCostHistory(double[]... histories) {
		
		Plot2DPanel plot = new Plot2DPanel();
		
		for (int i = 0; i < histories.length; i++)
			plot.addLinePlot("Cost " + i, histories[i]);
		
		JFrame frame = new JFrame("Costs");
        frame.setContentPane(plot);
        frame.setSize(600, 600);
        frame.setVisible(true);
		
	}
}
