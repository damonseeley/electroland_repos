public class QuoteLine extends TextBlock{
  
  // QUOTELINE.pde
  // works in clusters to override conventional animation/collision detection.
  
  private int quoteID;
  private int lineNumber;
  private Author author;           // author this quote came from
  private int introDelayDuration;  // duration to wait for textblocks to get pushed out of the way before appearing
  private int fadeInDuration;
  private int holdDuration;
  private int fadeOutDuration;
  private long startTime;
  private float trueWidth, trueHeight;
  private float quotationOffset;
  private float defaultAlphaVal = 255;
  private boolean alignRight = false;  // default aligned left
  private float pushMultiplier = 0.5;
  private float centerX, centerY;
  private float paragraphWidth, paragraphHeight;
  private int leftMargin = 0;
  
  public QuoteLine(int id, int quoteID, int lineNumber, Author author, String textValue, float x, float y, String fontName, int fontSize, float textScale, float quotationOffset){
    super(id, textValue, x, y, fontName, fontSize, textScale);
    this.quoteID = quoteID;
    this.lineNumber = lineNumber;
    this.author = author;
    this.quotationOffset = quotationOffset;
    uppercase = false;
    xv = 0;
    yv = 0;
    if(!uppercase){
      trueWidth = w = textRender.getWidth(this.lowercaseTextValue) * textScale;
      trueHeight = h = textRender.getHeight(this.lowercaseTextValue) * textScale;
    }
    introDelay = true;
    counter = 0;
  }
  
  public void applyGravity(float xGrav, float yGrav, float xgravity, float ygravity){
    // ignore gravity
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks, float xMargin, float yMargin){
    // only be able to slide horizontally.
    // only push against other text blocks, never get pushed.
    this.xMargin = xMargin;
    this.yMargin = yMargin;
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){                      // if older than this block, check if there is an overlap
        //if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
        if((abs(x - b.getX()) < abs((w*0.5 + b.getWidth()*0.5) + (xMargin*2))) && (abs(y - b.getY()) < abs((h*0.5 + b.getHeight()*0.5) + (xMargin*2)))){
          
          //float xoverlap = abs(w*0.5 + b.getWidth()*0.5) - abs(x - b.getX());    // no margins
          //float yoverlap = abs(h*0.5 + b.getHeight()*0.5) - abs(y - b.getY());
          float xoverlap = abs((w*0.5)+xMargin + (b.getWidth()*0.5)+xMargin) - abs(x - b.getX());
          float yoverlap = abs((h*0.5)+yMargin + (b.getHeight()*0.5)+yMargin) - abs(y - b.getY());
          
          float thisXvel = xv;  // make copies as they'll be modified simultaneously
          float thisYvel = yv;  
          float otherXvel = b.getXVel();
          float otherYvel = b.getYVel();
          
          if(y > b.getY()){    // this is below other textblock
            if(xoverlap > yoverlap){
              //yv += yoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, 0 - yoverlap * pushMultiplier * verticalSpring * textScale);
            } else {
              //yv += xoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, 0 - xoverlap * pushMultiplier * verticalSpring * textScale);
            }
          } else {             // other textblock is below this
            if(xoverlap > yoverlap){
              //yv -= yoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, yoverlap * pushMultiplier * verticalSpring * textScale);
            } else {
              //yv -= xoverlap * 0.5 * verticalSpring * b.getScale();
              b.push(0, xoverlap * pushMultiplier * verticalSpring * textScale);
            }
          }
          
          if(x > b.getX()){    // this is to the right of the other textblock
            if(xoverlap > yoverlap){
             //xv += yoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(0 - yoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            } else {
              //xv += xoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(0 - xoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            }
          } else {             // textblock is to the right of this
            if(xoverlap > yoverlap){
              //xv -= yoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(yoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            } else {
              //xv -= xoverlap * 0.5 * horizontalSpring * b.getScale();
              b.push(xoverlap * pushMultiplier * horizontalSpring * textScale, 0);
            }
          }
          
        }
      }
    }
  }
  
  public void clearArea(ConcurrentHashMap textBlocks, float multiplier){
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() > id){
        if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
          if(b.getX() > x){
            float xoverlap = (x+(w*0.5)) - (b.getX()-(b.getWidth()*0.5));
            b.push((xoverlap / (w*0.5)) * multiplier, 0);
          } else {
            float xoverlap = (b.getX() + (b.getWidth()*0.5)) - (x-(w*0.5));
            b.push((xoverlap / (w*0.5)) * multiplier, 0);
          }
        }
      } 
    }
  }
  
  public void render(PGraphicsOpenGL pgl, int xoffset, int yoffset, boolean yflip){
    
    if(introDelay){
      float progress = counter / (float)introDelayDuration;
      //w = trueWidth * progress;
      if(counter == introDelayDuration){
        introDelay = false;
        fadeIn = true;
        counter = 0;
      }
    } else if(fadeIn){
      float progress = sin((counter / (float)fadeInDuration) * (PI/2));
      if(counter == fadeInDuration){
        fadeIn = false;
        hold = true;
        counter = 0;
      }
      c = color(redVal, greenVal, blueVal, alphaVal*progress);
      
      pushMatrix();
      if(yflip){
        translate(x-xoffset, (stageHeight-y) - yoffset, 0);
      } else {
        translate(x-xoffset, y-yoffset, 0);
      }
      rotateX(radians(180));// fix for text getting rendered upside down for some reason
      //c = color(redVal, greenVal, blueVal, alphaVal);
      textRender.setColor( red(c)/255, green(c)/255, blue(c)/255, alpha(c)/255);
      pgl.beginGL();
      if(uppercase){
        textRender.print(uppercaseTextValue, 0-(w*0.5), 0-(h*0.5), 0, textScale);
        //textRender.print( join(subset(uppercaseTextValue.split(""), 0 , numChars), ""), 0-(w*0.5),0-(h*0.5),0,textScale);
      } else {
        textRender.print(lowercaseTextValue, 0-(w*0.5), 0-(h*0.5), 0, textScale);
        //textRender.print( join(subset(lowercaseTextValue.split(""), 0 , numChars), ""), 0-(w*0.5),0-(h*0.5),0,textScale);
      }
      pgl.endGL();
      popMatrix();
      
    } else if(hold){
      float progress = counter / (float)holdDuration;
      //println(counter +" "+ holdDuration);
      if(counter == holdDuration){
        hold = false;
        fadeOut = true;
        counter = 0;
      }
      c = color(redVal, greenVal, blueVal, alphaVal);
      super.render(pgl, xoffset, yoffset, yflip);
    } else if(fadeOut){
      float progress = sin((counter / (float)fadeOutDuration) * (PI/2));
      if(counter == fadeOutDuration){
        fadeOut = false;
        remove = true;
        counter = 0;
      }
      c = color(redVal, greenVal, blueVal, 255 - (alphaVal*progress));
      super.render(pgl, xoffset, yoffset, yflip);
    } else {
      c = color(redVal, greenVal, blueVal, alphaVal);
      super.render(pgl, xoffset, yoffset, yflip);
    }
    
    if(!fadeOut && !alignRight){
      x = (((author.getX() - (author.getWidth()*0.5)) + (w*0.5)) + 1 - quotationOffset);  // stay left aligned with author name at all times
    } else if(!fadeOut && alignRight){
      x = ((author.getX() + (author.getWidth()*0.5)) + (w*0.5)) + leftMargin;
      y = (author.getY() + (author.getHeight()*0.5)) - (h*0.5);
    }
    
    counter++;
  }
  
  public void push(float xforce, float yforce){
    // ignore vertical pushes
    if(!introDelay){
      //xv += xforce; // ignore horizontal pushes until typing animation
    }
  }
  
  public Author getAuthor(){
    return author;
  }
  
  public void snapToRight(int leftMargin){
    alignRight = true;
    this.leftMargin = leftMargin;
  }
  
  public int getQuoteID(){
    return quoteID;
  }
  
  public int getLineNumber(){
    return lineNumber;
  }
  
  public void setIntroDelay(int introDelayDuration){
    this.introDelayDuration = introDelayDuration;
  }
  
  public void setFadeInDuration(int fadeInDuration){
    this.fadeInDuration = fadeInDuration;
  }
  
  public void setHoldDuration(int holdDuration){
    this.holdDuration = holdDuration;
  }
  
  public void setFadeOutDuration(int fadeOutDuration){
    this.fadeOutDuration = fadeOutDuration;
  }
  
  public void setPushMultiplier(float pushMultiplier){
    this.pushMultiplier = pushMultiplier;
  }
  
  public void setParagraphCenter(float centerX, float centerY){
    this.centerX = centerX;
    this.centerY = centerY;
  }
  
  public void setParagraphDimensions(float paragraphWidth, float paragraphHeight){
    this.paragraphWidth = paragraphWidth;
    this.paragraphHeight = paragraphHeight;
  }
  
  public float getParagraphCenterX(){
    return centerX;
  }
  
  public float getParagraphCenterY(){
    return centerY;
  }
  
  public float getParagraphWidth(){
    return paragraphWidth;
  }
  
  public float getParagraphHeight(){
    return paragraphHeight;
  }
  
}
