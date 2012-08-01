package net.electroland.fish.core;

import java.net.SocketException;

import net.electroland.presenceGrid.net.GridDetectorClient;
import net.electroland.presenceGrid.net.GridDetectorDataListener;
import net.electroland.presenceGrid.util.GridProps;

public class VisionListener implements GridDetectorDataListener {
	SpacialGrid grid;
	
	int gridWidth;
	int gridHeight;

	int xOffset;
	int yOffset;
	
	public VisionListener(SpacialGrid grid, int port, int gridWidth, int gridHeight, int xOffset, int yOffset) {
		
		this.grid = grid;
		
		this.gridWidth = gridWidth;
		this.gridHeight = gridHeight;
		
		this.xOffset = xOffset;
		this.yOffset = yOffset;
		
		System.out.println("vision listener " + port+ " " + gridWidth + " "  + gridHeight + " " + xOffset + " " + yOffset);
		

		int bytesNeeded = (int) Math.ceil(((double) (gridWidth * gridHeight)) / 8.0);
		
		System.out.println("GridDetectorClient receiving " + bytesNeeded + " bytes per frame");
		
		try {
			new GridDetectorClient(port, bytesNeeded, this).start();
		} catch (SocketException e) {
			e.printStackTrace();
		}

	}

	public void receivedData(byte[] data) {
		int curBit = 128;
		int curByte = -1;
		
		

		

		int xStart = xOffset;
		int xStop = gridWidth + xOffset;
		int yStart = yOffset;
		int yStop = yOffset + gridHeight;
		
		for(int x = xStart; x < xStop; x++) { 
			SpacialGrid.Cell col[] = grid.cells[x];
			for(int y = yStart; y < yStop; y++) {
				if(curBit == 128) {
					curBit = 1;
					curByte++;
				} else {
					curBit = curBit <<1;
				}
				if((data[curByte] & curBit) != 0) {
					col[y].isTouched = true;
				} else {
					col[y].isTouched = false;					
				}
			}
		}	}

}
