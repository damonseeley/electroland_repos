package net.electroland.presenceGrid.net;

import java.awt.image.Raster;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;

import net.electroland.elvis.imaging.ThreshClamp;
import net.electroland.elvis.imaging.acquisition.axisCamera.AxisCamera;
import net.electroland.elvis.imaging.acquisition.jmyron.WebCam;
import net.electroland.presenceGrid.core.GridDetector;
import net.electroland.presenceGrid.core.GridListener;
import net.electroland.presenceGrid.ui.GridFrame;
import net.electroland.presenceGrid.util.GridProps;


public class GridDetectorServer implements GridListener {
	byte[] data;
	UdpServer server;
	boolean smooth;
	boolean[][] bDataGrid;

	public GridDetectorServer(String clientip, int port) throws SocketException, UnknownHostException {

		System.out.println("GridDetector sending to " + clientip + ":" + port);

		int gridWidth = GridProps.getGridProps().getProperty("gridWidth", 65);
		int gridHeight =GridProps.getGridProps().getProperty("gridHeight", 45);

		System.out.println("GridDetector grid dimensions " + gridWidth + "x" + gridHeight);

		int bytesNeeded = (int) Math.ceil(((double) (gridWidth * gridHeight)) / 8.0);

		System.out.println("GridDetector sending " + bytesNeeded + " bytes per frame");

		data = new byte[bytesNeeded] ;
		server = new UdpServer(clientip, port, data);

		GridDetector gridDetector;
		
		if(GridProps.getGridProps().getProperty("showGraphics", true)) {	
			GridFrame gf = new GridFrame(GridProps.getGridProps().getProperty("windowName", "EL Vision"));	
			gridDetector = gf.getGridPanel().gridDetector;
		} else {
			System.out.println("**** running vision headless *****");
			int srcImageWidth = GridProps.getGridProps().getProperty("srcImageWidth", 240);
			int srcImageHeight = GridProps.getGridProps().getProperty("srcImageHeight", 180);

			gridDetector = new GridDetector(srcImageWidth, srcImageHeight, gridWidth, gridHeight);
			gridDetector.createSetWarpGrid();

			gridDetector.setThresh(GridProps.getGridProps().getProperty("threshold", 5000.0));
			gridDetector.setBackgroundAdaptation(GridProps.getGridProps().setProperty("adaptation", .001));
			

			String ip = GridProps.getGridProps().getProperty("axisIP", "10.0.1.90");		
			String url = "http://" + ip + "/";
			String username = GridProps.getGridProps().getProperty("axisUsername", "root");
			String password = GridProps.getGridProps().getProperty("axisPassword", "n0h0");
			AxisCamera srcStream = new AxisCamera(url, srcImageWidth, srcImageHeight, 0, 0 , username, password, gridDetector);
			srcStream.start();
			gridDetector.start();
		}


		gridDetector.setGridListener(this);

		smooth = GridProps.getGridProps().getProperty("smoothDetection", false);
		bDataGrid = new boolean[gridWidth+2][gridHeight+2]; 
		
		System.out.println("smoothing " + smooth);

	}



	public void dataUpdateSmooth(Raster r) {
		for(int x = 1; x <= r.getWidth(); x++) { 
			boolean[] col = bDataGrid[x];
			for(int y = 1; y <= r.getHeight(); y++) {
				col[y] = r.getSampleDouble(x-1, y-1, 0) == ThreshClamp.WHITE;
			} 
		}

		int curBit = 128;
		int curByte = -1;



		for(int x = 1; x <= r.getWidth(); x++) { 
			for(int y = 1; y <= r.getHeight(); y++) {
				if(curBit == 128) {
					curBit = 1;
					curByte++;
					data[curByte] = 0;
				} else {
					curBit = curBit <<1;
				}

				if(bDataGrid[x][y] &&
						(bDataGrid[x-1][y-1] && bDataGrid[x][y-1] && bDataGrid[x-1][y]) ||
						(bDataGrid[x+1][y+1] && bDataGrid[x][y+1] && bDataGrid[x+1][y]) ||
						(bDataGrid[x-1][y+1] && bDataGrid[x][y+1] && bDataGrid[x-1][y]) ||
						(bDataGrid[x+1][y-1] && bDataGrid[x][y-1] && bDataGrid[x+1][y]) )

				{
					data[curByte] |= curBit;

				}
			}
		}
		try {
			server.send();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void dataUpdate(Raster r) {
		if(smooth) {
			dataUpdateSmooth(r);
			return;
		} 
		int curBit = 128;
		int curByte = -1;


		for(int x = 0; x < r.getWidth(); x++) { 
			for(int y = 0; y < r.getHeight(); y++) {
				if(curBit == 128) {
					curBit = 1;
					curByte++;
					data[curByte] = 0;
				} else {
					curBit = curBit <<1;
				}
				if(r.getSampleDouble(x, y, 0) == ThreshClamp.WHITE) {
					data[curByte] |= curBit;
				} 
			}
		}
		try {
			server.send();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static void main(String arg[]) throws IOException {
		if(arg.length > 0) {
			GridProps.init(arg[0]);
		} else {
			GridProps.init("gridProps.props");
		}

		new GridDetectorServer(
				GridProps.getGridProps().getProperty("clientIp", "localhost"),
				GridProps.getGridProps().getProperty("port", 1492)
		);




	}
}
