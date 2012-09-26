package net.electroland.gotham.processing.assets;

/**
 * Generic timer used for animations.
 * @author Aaron Seigal, modified with permission by Michael Kontopoulos
 */
public class Timer {

   private long duration;
   private long startTime;
   private boolean stopped = false;
   private boolean paused = false;
   private float pauseValue;

   public Timer(final long duration) {
      this.duration = duration;
   }

   /**
    * @return True if duration has elapsed.
    */
   public boolean isFinished() {
      if (System.currentTimeMillis() - startTime > duration) {
    	  stopped = true;
         return true;
      }
      return false;
   }
   
   public void pause(){
	   paused = true;
	   stopped = false;
	   pauseValue = this.progress();
   }

   /**
    * Start the timer from this point in time.
    */
   public void start() {
      startTime = System.currentTimeMillis();
      stopped = false;
      paused = false;
   }
   
   public void stop(){
	   stopped = true;
	   paused = false;
   }
   
   public void reset(final long duration){
	   this.duration = duration;
	   this.start();
   }

   /**
    * @return Value between 0-1 indicating progress of timer.
    */
   public float progress() {
	   if(stopped){
		   return 0;
	   } else if(paused){
		   return pauseValue;
	   }
      return (System.currentTimeMillis() - startTime) / (float) duration;
   }
   
   
   /**
    * Ramps up from start all the way to target.
    * @return Value between 0-1 indicating progress of timer.
    */
   public float rampProgress(){
	   if(stopped){
		   return 0;
	   } else if(paused){
		   return pauseValue;
	   }
	   return 1 - (float)Math.sin((Math.PI/2) + ((Math.PI/2) * this.progress()));
   }
   
   /**
    * Sinusoidal progress value. Ramps up from start and ramps down to target.
    * @return Value between 0-1 indicating progress of timer.
    */
   public float sinProgress(){
	   if(stopped){
		   return 0;
	   } else if(paused){
		   return pauseValue;
	   }
	   return (1 - ((float)Math.cos((Math.PI * this.progress())) / 2 - 0.5f)) - 1;
   }

}
