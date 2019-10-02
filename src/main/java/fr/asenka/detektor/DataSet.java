package fr.asenka.detektor;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;

import fr.asenka.detektor.util.Matrix;
import fr.asenka.detektor.util.MnistReader;

public class DataSet {

	private final String imagesFilePath;
	
	private final String labelsFilePath;
	
	private int size;
	
	private int[] labels;
	
	private List<int[][]> images;
	
	private int imageHeight;
	
	private int imageWidth;
	
	/**
	 * 
	 * @param imagesFilePath
	 * @param labelsFilePath
	 */
	public DataSet(String imagesFilePath, String labelsFilePath) {
		this.imagesFilePath = imagesFilePath;
		this.labelsFilePath = labelsFilePath;
		this.initialize();
	}
	
	/**
	 * 
	 */
	private void initialize() {
		
		try {
			labels = MnistReader.getLabels(Paths.get(labelsFilePath));
			images = MnistReader.getImages(Paths.get(imagesFilePath));
		} catch (IOException e) {
			throw new IllegalArgumentException(e);
		}
		
		if (images.isEmpty() || ArrayUtils.isEmpty(labels))
			throw new IllegalArgumentException("Labels or images is empty");
		else {
			imageHeight = images.get(0).length;
			imageWidth = images.get(0)[0].length;
		}

		if (labels.length != images.size())
			throw new IllegalArgumentException("Labels and images sizes are different");
		else 
			size = labels.length;
	}
	
	public int size() {
		return size;
	}
	
	public int getLabel(int index) {
		return this.labels[index];
	}
	
	public int[] getLabels() {
		return labels;
	}

	public Matrix getImage(int index) {
		return new Matrix(this.images.get(index));
	}
	
	public Matrix getVectorizedImage(int index) {
		return getImage(index).flatRow();
	}
	
	public Matrix getVectorizedImages() {
		
		Matrix result = new Matrix(size, imageHeight * imageWidth);
		
		for (int i = 0; i < size; i++)
			result.setRow(i, new Matrix(images.get(i)).flatRow());
		
		return result;
	}
	
	public Matrix getVectorizedLabels() {
		
		Matrix result = new Matrix(size, 1);
		
		for (int i = 0; i < size; i++)
			result.set(i, 0, labels[i]);
		
		return result;
	}
}
