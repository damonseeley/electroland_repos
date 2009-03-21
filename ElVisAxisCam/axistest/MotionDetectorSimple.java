package axistest;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;


public class MotionDetectorSimple{
	

	BufferedImage srcImg;
	BufferedImage bgModelImg;

	
	public MotionDetectorSimple(BufferedImage srcimg) {
		this.srcImg = srcimg;
		//init();
	}
	
	public void init() {
		//setup the bgModelImg so we have a base to compare to
		//temp simple clone
		cloneSrc();
	}
	
	public void process(BufferedImage srcimg) {
		srcImg = srcimg;
		
		compare();
		cloneSrc();
	}
	
	public void cloneSrc() {
		bgModelImg = new BufferedImage(srcImg.getWidth(), srcImg.getHeight(), srcImg.getType());
		Graphics2D g2D = bgModelImg.createGraphics();
		g2D.drawImage(srcImg, 0, 0, null);
	}
	
	public void compare() {
		int totalDiff = 0;
		if (srcImg != null && bgModelImg != null){
			for (int y = 0; y < srcImg.getHeight(); y++) {
				for (int x = 0; x < srcImg.getWidth(); x++) {

					int gray1 = srcImg.getRGB(x,y);
					int gray2 = bgModelImg.getRGB(x,y);
					int diff = gray1-gray2;
					totalDiff += Math.abs(diff);

				}
			}
		}
		
		double possDiff = srcImg.getHeight() * srcImg.getWidth() * 16777216 +.0001;
		if (totalDiff > 0 ){
			double diffPercentage = (totalDiff/possDiff) * 100;
			if (diffPercentage > 0){
				System.out.println(diffPercentage);
			}
		}
		
		

		
		
		

	}
	
}