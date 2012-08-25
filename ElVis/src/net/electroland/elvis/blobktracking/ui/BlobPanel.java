package net.electroland.elvis.blobktracking.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.AffineTransform;
import java.awt.image.RenderedImage;
import java.io.IOException;
import java.util.Vector;

import javax.swing.JPanel;
import javax.swing.Timer;

import net.electroland.elvis.blobktracking.core.BlobTracker;
import net.electroland.elvis.blobtracking.Blob;
import net.electroland.elvis.blobtracking.Track;
import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.blobtracking.TrackResults;
import net.electroland.elvis.imaging.PresenceDetector.ImgReturnType;
import net.electroland.elvis.manager.ImagePanel;
import net.electroland.elvis.util.ElProps;


public class BlobPanel extends JPanel implements ActionListener {




	public SimpleTrackListener[] trackListeners;



	AffineTransform imageScaler;


	ElProps props ;

	public BlobTracker blobTracker;

	public BlobPanel(ElProps props, BlobTracker blobTracker) {
		this.props =props;
		this.blobTracker = blobTracker;		
		this.blobTracker.presenceDetector.setImageReturn(ImgReturnType.CONTOUR);
		/*
		try {
			this.blobTracker.setSourceStream(ImagePanel.FLY_SRC);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 */







		Timer t = new Timer( (int) (12.0/1000.0), this);
		t.setInitialDelay(1000);
		t.start();

		setSize(blobTracker.presenceDetector.getWidth(), blobTracker.presenceDetector.getHeight());
		setPreferredSize(new Dimension(blobTracker.presenceDetector.getWidth(), blobTracker.presenceDetector.getHeight()));


	}





	public void paint(Graphics g) {
		super.paint(g);
		Graphics2D g2d = (Graphics2D)g;

		RenderedImage ri = blobTracker.presenceDetector.getBufferedImage();

		if (ri != null) {
			g2d.drawRenderedImage(ri,  imageScaler);
		}

		renderDrawing(g2d);
	}

	public void renderDrawing(Graphics2D g2d) {
		Vector<Blob> blobs = blobTracker.presenceDetector.getBlobs();
		boolean drawTracks = props.getProperty("drawTracks", true);

		for(Blob b :blobs) {
			if (drawTracks) {
				b.paint(g2d);
			}
		}
		if(drawTracks) {
			for(SimpleTrackListener tl: trackListeners) {
				Vector<Track> trackCache = tl.tracks;
				for(Track t : trackCache) {
					if (props.getProperty("drawTracks", true)) {
						//System.out.println("tracks");
						t.paint(g2d);
					}
				}			
			}
		}
	}


	public void actionPerformed(ActionEvent e) {
		repaint();				
	}

	public void stop() {
		blobTracker.stopRunning();
	}




	public static class SimpleTrackListener implements TrackListener {
		Vector<Track> tracks = new Vector<Track>();

		public void updateTracks(TrackResults results) {
			this.tracks = results.existing;
		}
	}

}
