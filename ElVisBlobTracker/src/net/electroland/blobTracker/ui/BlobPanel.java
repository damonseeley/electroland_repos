package net.electroland.blobTracker.ui;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.AffineTransform;
import java.awt.image.RenderedImage;
import java.util.Vector;

import javax.swing.JPanel;
import javax.swing.Timer;

import net.electroland.blobDetection.Blob;
import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.core.BlobTracker;
import net.electroland.blobTracker.util.ElProps;


public class BlobPanel extends JPanel implements ActionListener {




	public SimpleTrackListener[] trackListeners;



	AffineTransform imageScaler;


	ElProps props = ElProps.THE_PROPS;

	public BlobTracker blobTracker;

	public BlobPanel(BlobTracker blobTracker) {

		this.blobTracker = blobTracker;





		Timer t = new Timer( (int) (12.0/1000.0), this);
		t.setInitialDelay(1000);
		t.start();

		setSize(blobTracker.w, blobTracker.h);
		setPreferredSize(new Dimension(blobTracker.w, blobTracker.h));


	}





	public void paint(Graphics g) {
		super.paint(g);
		Graphics2D g2d = (Graphics2D)g;

		RenderedImage ri = blobTracker.getImage();

		if (ri != null) {
			g2d.drawRenderedImage(ri,  imageScaler);
		}

		renderDrawing(g2d);
	}

	public void renderDrawing(Graphics2D g2d) {
		for(Blob b :blobTracker.newFrameBlobs) {
			if (props.getProperty("drawTracks", true)) {
				b.paint(g2d);
			}
		}

		for(SimpleTrackListener tl: trackListeners) {
			Vector<Track> trackCache = tl.tracks;
			for(Track t : trackCache) {
				if (props.getProperty("drawTracks", true)) {
					t.paint(g2d);
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
