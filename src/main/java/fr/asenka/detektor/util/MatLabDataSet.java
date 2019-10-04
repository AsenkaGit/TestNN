package fr.asenka.detektor.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public class MatLabDataSet {
	
	private Matrix X;
	
	private Matrix y;
	
	private Matrix theta1;
	
	private Matrix theta2;
	
	public MatLabDataSet() throws IOException {

		MatFileReader dataReader = new MatFileReader("C:\\works\\tests\\neuralNetwork\\detektor\\src\\main\\resources\\ex4data1.mat");
//		MatFileReader weightsReader = new MatFileReader("C:\\works\\tests\\neuralNetwork\\detektor\\src\\main\\resources\\ex4weights.mat");
		MatFileReader weightsReader = new MatFileReader("C:\\works\\tests\\neuralNetwork\\detektor\\src\\main\\resources\\output.mat");
		
		MLDouble ML_X = (MLDouble) dataReader.getMLArray("X");
		MLDouble ML_y = (MLDouble) dataReader.getMLArray("y");
		MLDouble ML_theta1 = (MLDouble) weightsReader.getMLArray("Theta1");
		MLDouble ML_theta2 = (MLDouble) weightsReader.getMLArray("Theta2");
		
		X = new Matrix(ML_X.getArray());
		y = new Matrix(ML_y.getArray()).subtract(1);
		theta1 = new Matrix(ML_theta1.getArray());
		theta2 = new Matrix(ML_theta2.getArray());
	}

	public Matrix getImages() {
		return X;
	}

	public Matrix getLabels() {
		return y;
	}

	public Matrix getTheta1() {
		return theta1;
	}

	public Matrix getTheta2() {
		return theta2;
	}
	
	public static void saveWeights(Matrix theta1, Matrix theta2) throws IOException {
		
		MatFileWriter writer = new MatFileWriter();
		
		MLArray ML_theta1 = new MLDouble("Theta1", theta1.getRawData());
		MLArray ML_theta2 = new MLDouble("Theta2", theta2.getRawData());
		
		Collection<MLArray> data = new ArrayList<>();
		data.add(ML_theta1);
		data.add(ML_theta2);
		
		writer.write("C:\\works\\tests\\neuralNetwork\\detektor\\src\\main\\resources\\output.mat", data);
	}
}
