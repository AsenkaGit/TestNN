package fr.asenka.detektor.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class MatrixTest {
	
	private final double DELTA = 0.0000001d;

	@Test
	void testNewMatrix() {
		
		Matrix m = new Matrix("1 2 3 ; 4 5 6 ; 7 8 9");
		
		assertEquals(3, m.getRows());
		assertEquals(3, m.getColumns());
		assertEquals(1d, m.get(0, 0), DELTA);
		assertEquals(2d, m.get(0, 1), DELTA);
		assertEquals(3d, m.get(0, 2), DELTA);
		assertEquals(4d, m.get(1, 0), DELTA);
		assertEquals(5d, m.get(1, 1), DELTA);
		assertEquals(6d, m.get(1, 2), DELTA);
		assertEquals(7d, m.get(2, 0), DELTA);
		assertEquals(8d, m.get(2, 1), DELTA);
		assertEquals(9d, m.get(2, 2), DELTA);
		
		Matrix p = new Matrix(new double[][] {{1d, 2d, 3d}, {4d, 5d, 6d}, {7d, 8d, 9d}});
		
		assertEquals(p, m);
	}
	
	@Test
	void testTranspose() {
		
		Matrix m = new Matrix("1 2 3 ; 4 5 6");
		Matrix expected = new Matrix("1 4 ; 2 5 ; 3 6");
		
		assertEquals(expected, m.transpose());
	}
	
	@Test
	void testMultiply() {
		
		Matrix m1 = new Matrix("1 2 3 ; 4 5 6");
		Matrix m2 = new Matrix("2 4 ; 3 0 ; 1 0");
		Matrix expectedM1xM2 = new Matrix("11 4 ; 29 16");
		Matrix expectedM2xM1 = new Matrix("18 24 30 ; 3 6 9 ; 1 2 3");
		
		assertEquals(expectedM1xM2, m1.multiply(m2));
		assertEquals(expectedM2xM1, m2.multiply(m1));
	}

}
