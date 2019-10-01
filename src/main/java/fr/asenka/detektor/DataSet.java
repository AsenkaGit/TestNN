package fr.asenka.detektor;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import fr.asenka.detektor.util.MnistReader;

public class DataSet {

	private final String imagesFilePath;
	
	private final String labelsFilePath;
	
	private int size;
	
	private int[] labels;
	
	private List<int[][]> images;
	
	private int numClasses;
	
	private int imageHeight;
	
	private int imageWidth;
	
	/**
	 * 
	 * @param imagesFilePath
	 * @param labelsFilePath
	 */
	public DataSet(String imagesFilePath, String labelsFilePath, int numClasses) {
		this.imagesFilePath = imagesFilePath;
		this.labelsFilePath = labelsFilePath;
		this.numClasses = numClasses;
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
	
	/**
	 * 
	 * @return
	 */
	public int size() {
		return size;
	}
	
	/**
	 * 
	 * @param index
	 * @return
	 */
	public int[][] getImage(int index) {
		return this.images.get(index);
	}
	
	/**
	 * 
	 * @param index
	 * @return
	 */
	public int getLabel(int index) {
		return this.labels[index];
	}
	
	public int[] getLabels() {
		return labels;
	}
	
	/**
	 * 
	 * @param index
	 * @return
	 */
	public RealMatrix getVectorizedLabel(int index) {
		
		double[] values = new double[numClasses];
		
		for (int k = 0; k < numClasses; k++)
			values[k] = k == labels[index] ? 1d : 0d;
		
		return MatrixUtils.createColumnRealMatrix(values);
	}
	
	/**
	 * 
	 * @return
	 */
	public RealMatrix getVectorizedLabels() {
		
		double[][] values = new double[size][numClasses];
		
		for (int index = 0; index < size; index++)
			for(int k = 0; k < numClasses; k++)
				values[index][k] = k == labels[index] ? 1d : 0d;
		
		return MatrixUtils.createRealMatrix(values);
	}
	
	/**
	 * 
	 * @param index
	 * @return
	 */
	public RealMatrix getVectorizedImage(int index) {
		int[][] image = images.get(index);
		double[] values = flat(image);
		return MatrixUtils.createColumnRealMatrix(values);
	}
	
	/**
	 * 
	 * @param index
	 * @return
	 */
	public RealMatrix getScaledAndVectorizedImage(int index) {
		
		int[][] image = images.get(index);
		double[] values = scale(flat(image));
		return MatrixUtils.createColumnRealMatrix(values);
	}
	
	/**
	 * 
	 * @return
	 */
	public RealMatrix getVectorizedImages() {
		
		double[][] values = new double[images.size()][];
		
		for (int i = 0; i < images.size(); i++)
			values[i] = flat(images.get(i));
		
		return MatrixUtils.createRealMatrix(values);
	}
	
	/**
	 * 
	 * @return
	 */
	public RealMatrix getScaledAndVectorizedImages() {
		
		double[][] values = new double[images.size()][];
		
		for (int i = 0; i < images.size(); i++)
			values[i] = scale(flat(images.get(i)));
		
		return MatrixUtils.createRealMatrix(values);
	}
	
	/**
	 * 
	 * @param image
	 * @return
	 */
	private double[] flat(int[][] image) {
		
		double[] flatImage = new double[imageWidth * imageHeight];
		
		for (int i = 0; i < image.length; i++)
			for (int j = 0; j < image[i].length; j++) 
				flatImage[imageWidth * i + j] = (double) image[i][j];
			
		return flatImage;
	}
	
	/**
	 * 
	 * @param image
	 * @return
	 */
	private double[] scale(double[] image) {
		
		double[] scaledImage = new double[imageWidth * imageHeight];
		
		for (int i = 0; i < image.length; i++)
			scaledImage[i] = image[i] / 255d; // x[i] = (x[i] - min(x)) / (max(x) - min(x))
		
		return scaledImage;
	}
}
