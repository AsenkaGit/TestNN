package fr.asenka.detektor.util;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.nio.ByteBuffer;

public final class MnistReader {

	public static final int[] getLabels(Path labelsFile) throws IOException {
		
		ByteBuffer buffer = ByteBuffer.wrap(decompressGzip(Files.readAllBytes(labelsFile)));
		
		if (buffer.getInt() != 2049) 
			throw new IOException("Not a labels file");
		
		int numLabels = buffer.getInt();
		int[] labels = new int[numLabels];
		
		for (int i = 0; i < numLabels; i++)
			labels[i] = buffer.get() & 0xFF;
		
		return labels;
	}
	
	public static final List<int[][]> getImages(Path imagesFile) throws IOException {
		
		ByteBuffer buffer = ByteBuffer.wrap(decompressGzip(Files.readAllBytes(imagesFile)));
		
		if (buffer.getInt() != 2051)
			throw new IOException("Not an images file");
		
		int numImages = buffer.getInt();
		int numRows = buffer.getInt();
		int numColumns = buffer.getInt();
		
		List<int[][]> images = new ArrayList<>();
		
		for (int i = 0; i < numImages; i++) {	
			int[][] image = new int[numRows][];
			
			for (int row = 0; row < numRows; row++) {
				image[row] = new int[numColumns];
				
				for (int col = 0; col < numColumns; col++)
					image[row][col] = buffer.get() & 0xFF;
			}
			images.add(image);
		}
		return images;
	}

	private static final byte[] decompressGzip(final byte[] gzipData) throws IOException {
		
		try(ByteArrayInputStream stream = new ByteArrayInputStream(gzipData); 
			GZIPInputStream gzipStream = new GZIPInputStream(stream); 
			ByteArrayOutputStream output = new ByteArrayOutputStream()) {
			
			byte[] buffer = new byte[8192];
			int n;
			
			while((n = gzipStream.read(buffer)) > 0)
				output.write(buffer, 0, n);
			
			return output.toByteArray();
		}
	}
}
