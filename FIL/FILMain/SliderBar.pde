public class SliderBar extends Widget{
  
  private float clickX;
  private float totalWidth;
  private boolean dragging = false;
  private PImage segmentImg, leftImg, rightImg;
  private PImage segmentImgDown, leftImgDown, rightImgDown;
  
  public SliderBar(String name, float x, float y, float w, float h, float value, PImage segmentImg, PImage segmentImgDown,
                   PImage leftImg, PImage leftImgDown, PImage rightImg, PImage rightImgDown){
    super(name, x, y, w, h, value);
    this.totalWidth = w;
    this.segmentImg = segmentImg;
    this.leftImg = leftImg;
    this.rightImg = rightImg;
    this.segmentImgDown = segmentImgDown;
    this.leftImgDown = leftImgDown;
    this.rightImgDown = rightImgDown;
  }
  
  public void dragged(){
    // TODO: move bar location along slider and continually send widgetEvents giving normalized offset value
    if(x >= 0 && x+w <= totalWidth){
      x = mouseX + clickX;
      if(x < 0){
        x = 0;
      } else if(x+w > totalWidth){
        x = totalWidth - w;
      }
      WidgetEvent newwe = new WidgetEvent(this, DRAGGED, true);
      super.newEvent(newwe);
    } 
  }
  
  public void draw(){
    //noStroke();
    //fill(50,150,255,128);
    //rect(x,y,w,h);
    if(dragging){
      image(leftImgDown, x, y);
      image(segmentImgDown, x + leftImg.width, y, int(w - (leftImg.width+rightImg.width)) + 1, segmentImg.height);
      image(rightImgDown, int(x + leftImg.width + (w - (leftImg.width+rightImg.width))), y);
    } else {
      image(leftImg, x, y);
      image(segmentImg, x + leftImg.width, y, int(w - (leftImg.width+rightImg.width)) + 1, segmentImg.height);
      image(rightImg, int(x + leftImg.width + (w - (leftImg.width+rightImg.width))), y);
    }
  }
  
  public void pressed(){
    clickX = x - mouseX;
    dragging = true;
  }
  
  public void released(){
    dragging = false;
  }
  
  public void rollOut(){
    // ignore
  }
  
  public void rollOver(){
    // ignore
  }
  
  public void cursorMovement(){
    // ignore
  }
  
  public void setWidth(float w){
    this.w = w;
    if(x < 0){
      x = 0;
    } else if(x+w > totalWidth){
      x = totalWidth - w;
    }
  }
  
  public void setOffset(float offset){
    this.x = offset;
    if(x < 0){
      x = 0;
    } else if(x+w > totalWidth){
      x = totalWidth - w;
    }
  }
  
}
