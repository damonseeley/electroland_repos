package axis;

import java.awt.image.BufferedImage;

public interface ImageReceiver {
	public void addImage(BufferedImage i);
	public void receiveErrorMsg(Exception cameraException);
}
