package net.electroland.elvis.blobtracking;

import java.util.Collection;
import java.util.HashSet;
import java.util.Vector;

public class Grid {


	float maxDist;
	float maxBlobDistSqr;

	int clusterSize = -1;

	float gridCellSize;
	float gridCellScaler;
	int gridWidth;
	int gridHeight;
	Cell[][] cells;
	HashSet<Blob> allBlobs = new HashSet<Blob>();


	public class Cell {
		Vector<Blob> blobs = new Vector<Blob>();
	}

	/**
	 * 
	 * @param width - world width
	 * @param height - world height
	 * @param maxBlobDist - max distance a track can move and be considered same object
	 */	
	
	public Grid(float width, float height, float maxBlobDist) {
		this.maxDist = maxBlobDist;
		this.maxBlobDistSqr = maxBlobDist * maxBlobDist;
		gridCellSize = maxBlobDist * .5f;
		gridCellScaler = 1.0f /gridCellSize;
		gridWidth = (int) Math.ceil( width * gridCellScaler);
		gridHeight = (int) Math.ceil( height * gridCellScaler);
		cells = new Cell[ gridWidth ][gridHeight];
		
		
		for(int x = 0; x < gridWidth; x++) {
			Cell[] col = cells[x];
			for(int y = 0; y < col.length; y++) {
				col[y] = new Cell();
			}
		}
	}

	public void clear() {
		for(int x = 0; x < gridWidth; x++) {
			Cell[] col = cells[x];
			for(int y = 0; y < col.length; y++) {
				col[y].blobs.clear();
			}
		}
		allBlobs.clear();
	}

	public Cell getCell(float x, float y) {
		return cells[(int) Math.floor(x * gridCellScaler)][(int) Math.floor(y * gridCellScaler)];		
	}

	
	public void addBlobs(Collection<Blob> blobs) {
		for(Blob b : blobs) {
			/* don't need to merge existing blobs anymore?
			if(b.getSize() < clusterSize) {
				HashSet<Blob> neighbors = this.getPossibleMatchs(b.centerX, b.centerY);
				HashSet<Blob> cluster = new HashSet<Blob>(neighbors);
				b.setAndUpdateCluster(cluster);
				for(Blob clusterBlob : neighbors) {
					clusterBlob.setAndUpdateCluster(cluster);
				}
			}
			*/
			addBlob(b);				
			
		}
	}
	
	/*
	public HashSet<Blob> mergedBlobs() {
		HashSet<Blob> newBlobs = new HashSet<Blob>(allBlobs.size());
		HashSet<Blob> discardedBlobs = new HashSet<Blob>(allBlobs.size());
		for(Blob b : allBlobs) {
			if(! discardedBlobs.contains(b)) {
				if(b.cluster != null) {
					Blob newBlob = new Blob();
					for(Blob clusterBlob : b.cluster) {
						newBlob.cluster(clusterBlob);
						discardedBlobs.add(clusterBlob);
					}
					discardedBlobs.add(b);
					newBlobs.add(newBlob);
				} else {
					newBlobs.add(b);
				}
			}
		}
		return newBlobs;
	}
	*/

	
	
	public void addBlob(Blob b) {
		getCell(b.centerX, b.centerY).blobs.add(b);
		allBlobs.add(b);
	}

	public static final int ALL = 1 | 2 | 4| 8;
	public static final int TOP = 1 | 2 | 8;
	public static final int BOTTOM = 1 | 2 | 4;
	public static final int RIGHT = 1 | 4 | 8;
	public static final int LEFT = 2 | 4 | 8;
	public static final int TOP_RIGHT = 1| 8;
	public static final int TOP_LEFT = 2| 8;
	public static final int BOTTOM_RIGHT = 1 |4;
	public static final int BOTTOM_LEFT = 2 | 4;


	// 1,0  ends up doing left hand having a problem (should be top)

	public HashSet<Blob> getPossibleMatchs(float x, float y) {
		HashSet<Blob> blobs = new HashSet<Blob>();

		int gridX = (int) Math.floor(x * gridCellScaler);
		int gridY = (int) Math.floor(y * gridCellScaler);
		
		gridX = (gridX < 0) ? 0 : gridX;
		gridX = (gridX >= gridWidth) ? gridWidth -1 : gridX;
		
		gridY = (gridY < 0) ? 0 : gridY;
		gridY = (gridY >= gridHeight) ? gridHeight -1 : gridY;
		

		


		blobs.addAll(cells[gridX][gridY].blobs);

		int location = (gridX > 0) ? 1 : 0;
		if (gridX < gridWidth -1) {
			location |= 2;
		}
		if(gridY > 0) {
			location |= 4;
		}
		if (gridY < gridWidth -1) {
			location |= 8;
		}
		try {

			switch(location) {
			case ALL:
				blobs.addAll(cells[gridX-1][gridY].blobs);
				blobs.addAll(cells[gridX+1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY-1].blobs);
				blobs.addAll(cells[gridX][gridY+1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY-1], blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY-1], blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY+1], blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY+1], blobs);
				break;
			case TOP:
				blobs.addAll(cells[gridX-1][gridY].blobs);
				blobs.addAll(cells[gridX+1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY+1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY+1], blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY+1], blobs);
				break;
			case BOTTOM:
				blobs.addAll(cells[gridX-1][gridY].blobs);
				blobs.addAll(cells[gridX+1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY-1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY-1], blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY-1], blobs);
				break;
			case LEFT:
				blobs.addAll(cells[gridX+1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY-1].blobs);
				blobs.addAll(cells[gridX][gridY+1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY-1], blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY+1], blobs);
				break;
			case RIGHT:
				blobs.addAll(cells[gridX-1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY-1].blobs);
				blobs.addAll(cells[gridX][gridY+1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY-1], blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY+1], blobs);
				break;
			case TOP_LEFT:
				blobs.addAll(cells[gridX+1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY+1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY+1], blobs);
				break;
			case TOP_RIGHT:
				blobs.addAll(cells[gridX-1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY+1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY+1], blobs);
				break;
			case BOTTOM_LEFT:
				blobs.addAll(cells[gridX+1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY-1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX+1][gridY-1], blobs);
				break;
			case BOTTOM_RIGHT:
				blobs.addAll(cells[gridX-1][gridY].blobs);
				blobs.addAll(cells[gridX][gridY-1].blobs);
				addWithinRangeAndSize(x,y, cells[gridX-1][gridY-1], blobs);
				break;
			}

		} catch (RuntimeException e) {
			System.out.println("(" + x + ", " + y + ") -> [" + gridX + "][" + gridY + "]   LOC:" + location );
			throw e;
		}


		return blobs;

	}

	public void setClusterSize(int size) {
		this.clusterSize = size;
	}

	private void addWithinRangeAndSize(float x, float y, Cell cell, Collection<Blob> blobs) {
		for(Blob b : cell.blobs) {
			if((b.getSize() < this.clusterSize) && (b.distSqr(x, y) <= maxBlobDistSqr)) {
				blobs.add(b);
			}
		}
	}

//	ALL          15
//	TOP          11
//	BOTTOM       7
//	RIGHT        13
//	LEFT         14
//	TOP_RIGHT    9
//	TOP_LEFT     10
//	BOTTOM_RIGHT 5
//	BOTTOM_LEFT  6


//	public static void main(String arg[]) {
//	System.out.println("ALL          "  + (1 | 2 | 4| 8));
//	System.out.println("TOP          "  + (1 | 2 | 8));
//	System.out.println("BOTTOM       "  + (1 | 2 | 4));
//	System.out.println("RIGHT        "  + (1 |   4| 8));
//	System.out.println("LEFT         "  + ( 2 | 4| 8));
//	System.out.println("TOP_RIGHT    "  + (1 |  8));
//	System.out.println("TOP_LEFT     "  + (2| 8));
//	System.out.println("BOTTOM_RIGHT "  + (1|4));
//	System.out.println("BOTTOM_LEFT  "  + (2 | 4));
//	}

}
