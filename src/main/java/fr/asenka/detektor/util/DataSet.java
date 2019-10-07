package fr.asenka.detektor.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public abstract class DataSet {

	protected Matrix X;
	
	protected Matrix y;
	
	protected DataSet() {
		loadData();
	}
	
	protected abstract void loadData();
	
	public abstract int getImageSize();
	
	public Matrix getImages() {
		return X;
	}

	public Matrix getLabels() {
		return y;
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
