package fr.asenka.detektor.util;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

public class MnistDataSet extends DataSet {

	private static final int IMAGE_SIZE = 28 * 28;
	
	@Override
	protected void loadData() {
		System.out.println("Loading MNIST data...");

		try {
			List<int[][]> images = MnistReader.getImages(Paths.get("src/main/resources/train-images-idx3-ubyte.gz"));
			int[] labels = MnistReader.getLabels(Paths.get("src/main/resources/train-labels-idx1-ubyte.gz"));
			
			X = prepareImagesMatrix(images);
			y = prepareLabelsMatrix(labels);

			System.out.println(X.rows() + " images loaded.");
			
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public int getImageSize() {
		return IMAGE_SIZE;
	}

	private static final Matrix prepareImagesMatrix(List<int[][]> images) {

		Matrix result = new Matrix(images.size(), IMAGE_SIZE);

		for (int i = 0; i < images.size(); i++) {
			Matrix image = new Matrix(images.get(i));
			result.setRow(i, image.flatRow());
		}
		return result.normalize();
	}

	private static final Matrix prepareLabelsMatrix(int[] labels) {

		Matrix result = new Matrix(labels.length, 1);

		for (int i = 0; i < labels.length; i++)
			result.set(i, 0, labels[i]);

		return result;
	}
}
