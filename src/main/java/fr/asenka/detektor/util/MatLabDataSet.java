package fr.asenka.detektor.util;

import java.io.IOException;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class MatLabDataSet extends DataSet {
	
	private static final int IMAGE_SIZE = 20 * 20;
	
	private Matrix theta1;
	
	private Matrix theta2;
	
	@Override
	protected void loadData() {

		System.out.println("Loading MatLab data...");
		
		try {
		MatFileReader dataReader = new MatFileReader("C:\\works\\tests\\neuralNetwork\\detektor\\src\\main\\resources\\ex4data1.mat");
		MatFileReader weightsReader = new MatFileReader("C:\\works\\tests\\neuralNetwork\\detektor\\src\\main\\resources\\ex4weights.mat");
//		MatFileReader weightsReader = new MatFileReader("C:\\works\\tests\\neuralNetwork\\detektor\\src\\main\\resources\\output.mat");
		
		MLDouble ML_X = (MLDouble) dataReader.getMLArray("X");
		MLDouble ML_y = (MLDouble) dataReader.getMLArray("y");
		MLDouble ML_theta1 = (MLDouble) weightsReader.getMLArray("Theta1");
		MLDouble ML_theta2 = (MLDouble) weightsReader.getMLArray("Theta2");
		
		X = new Matrix(ML_X.getArray());
		y = new Matrix(ML_y.getArray()).subtract(1);
		
		System.out.println(X.rows() + " images loaded.");
		
		theta1 = new Matrix(ML_theta1.getArray());
		theta2 = new Matrix(ML_theta2.getArray());
		
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public int getImageSize() {
		return IMAGE_SIZE;
	}

	public Matrix getTheta1() {
		return theta1;
	}

	public Matrix getTheta2() {
		return theta2;
	}
}
