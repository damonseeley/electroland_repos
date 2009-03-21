package net.electroland.blobDetection;

import java.awt.image.Raster;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

import net.electroland.blobTracker.util.RegionMap;
import net.electroland.elvis.imaging.ThreshClamp;

public class Blobs {
	RegionMap regionMap;

	Blob curBlob = null;
	int curBlobId = -1;
	int nextBlobId = 0;

	HashMap<Integer, Blob> blobHash = new HashMap<Integer, Blob>();
	Vector<Blob>[] detectedBlobs;

	int[][] blobPixels = null;


	public Vector<Blob> getDetectedBlobs(int region) {
		return detectedBlobs[region];
	}

	public Blobs(int srcWidth, int srcHeight, RegionMap regionMap) {
		blobPixels = new int[srcHeight][srcWidth];
		this.regionMap = regionMap;
		detectedBlobs = new Vector[regionMap.size()];
		for(int i = 0; i < regionMap.size(); i++) {
			detectedBlobs[i]= new Vector<Blob>();
		}
		
	}


	public void calcAndCullBlobs() {
		for(Vector<Blob> vec : detectedBlobs) {
			vec.clear();
		}

		Iterator<Blob> i = blobHash.values().iterator();
		while(i.hasNext()) {
			Blob b = i.next();
			if(! b.centerIsCalculated) {
				b.calcCenter();
			}			
			Region r = regionMap.getRegion((int)b.centerX, (int) b.centerY);
			if((b.getSize() >= r.minBlobSize) && (b.getSize() <= r.maxBlobSize)) {
				detectedBlobs[r.id].add(b);
			}
		}

	}



	public  void detectBlobs(Raster data) {
		blobHash.clear();
		curBlob = null;

		if(data.getSampleDouble(0,0, 0) == ThreshClamp.WHITE) {
			createCurBlob();
			blobPixels[0][0] = curBlobId;
			curBlob.addPoint(0, 0);
		}

		int[] blobRow = blobPixels[0];
		for(int x = 0; x < data.getWidth(); x++) {
			if(data.getSample(x, 0, 0) == ThreshClamp.WHITE) {
				if(curBlob == null) {
					createCurBlob();
				}
				blobRow[x] = curBlobId;
				curBlob.addPoint(x, 0);
			} else {
				curBlob = null;
				blobRow[x] = -1;
			}
		}

		int[] lastBlobRow;

		for(int y = 1; y < data.getHeight(); y++) {
			lastBlobRow = blobRow;
			blobRow = blobPixels[y];
			curBlob = null;

			for(int x = 0; x < data.getWidth(); x++) {
				if(data.getSample(x, y, 0) == ThreshClamp.WHITE) {
					if(curBlob == null) {
						if(lastBlobRow[x] != -1) {
							curBlobId = lastBlobRow[x];
							curBlob = blobHash.get(curBlobId);
						} else {
							createCurBlob();
						}
					} else {
						if(lastBlobRow[x] != -1) { // merge
							curBlobId = lastBlobRow[x];
							Blob topBlob = blobHash.get(curBlobId);
							if(topBlob != curBlob) {
								topBlob.merger(curBlob);
								curBlob = topBlob;
								Iterator<Integer> e = topBlob.ids.iterator();
								while(e.hasNext()) {
									blobHash.put(e.next(), topBlob);
								}
							}

						} 
					}
					blobRow[x] = curBlobId;
					curBlob.addPoint(x, y);
				} else {
					curBlob = null;
					blobRow[x] = -1;
				}
			}
		}
		calcAndCullBlobs();

	}


	private void createCurBlob() {
		curBlob = new Blob();
		curBlobId = nextBlobId++;
		curBlob.ids.add(curBlobId);
		blobHash.put(curBlobId, curBlob);		
	}











}
