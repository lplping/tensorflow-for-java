package lp;
import java.awt.Shape;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import org.apache.commons.io.FileUtils;  
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class test {
	
	private static Map <String,Integer> map=new HashMap<String,Integer>();
	 
	//一、###################################
	//读取测试文件放入list中
	 public static Tensor transformer(String str, Map<String,Integer> char_to_id)
	 {  
		char[] chars=str.toCharArray();
		int[] input_data=new int[chars.length]; //转化成索引后的输入数据
		int[] final_input=new int[20];//输入数据
		int[][] input=new int[1][20];
		
		for (int k=0;k<chars.length;k++){
			if (char_to_id.containsKey(String.valueOf(chars[k]))){
				int index=(int)char_to_id.get(String.valueOf(chars[k]));//获取索引值
				input_data[k]=index;
			}
			else
			{
				int index=char_to_id.get(String.valueOf("<UNK>"));//获取索引值
				input_data[k]=index;
			}
		}
		
      //在后面补 0
		
		if (input_data.length>20){
			for (int i = 0;i < 20;i++) {
			    //System.out.print(input_data[i]+" ");
			    final_input[i]=input_data[i];
			}
		}
		else{
			
			for (int i = 0;i < input_data.length;i++) {
				final_input[i]=input_data[i];
			}
		    for (int i =input_data.length;i <20 ;i++) {
					final_input[i]=0;
			    //System.out.print(input_data[i]+" ");
			    
			}
		}
		
		for (int i=0;i<final_input.length;i++){
			input[0][i]=final_input[i];
			//System.out.println(input[0][i]);
		   
		}
		
		Tensor T=Tensor.create(input);
		return T;
		
		
		
	}
	//读取词典文件list中
		
	 public static void  readdict()
	 {  
		 List<String> s=new ArrayList();
		 try{
			 s=FileUtils.readLines(new File("d:/a/dict.txt"),"utf-8");  
			 for (String str: s){
				 String key=str.split(":")[0];
				 //System.out.println(key) ;	
				 int  value=Integer.parseInt(str.split(":")[1]);
				 //System.out.println(value) ;	
				 map.put(key, value); 
			 }
			//		 
		 }
		 catch(Exception e){
			 e.printStackTrace();
		 }    
	    
	}
	 private static byte[] readAllBytesOrExit(Path path) {
         try {
             return Files.readAllBytes(path);
         } catch (IOException e) {
             System.err.println("Failed to read [" + path + "]: "
                     + e.getMessage());
             System.exit(1);
         }
         return null;
     }
	 public static void main(String[] args) throws Exception {
		 readdict();
		 //读数据  
	           
	            try {   
	            	InputStreamReader isr = new InputStreamReader(new FileInputStream("d:/a/temp.txt"), "UTF-8");
	            	BufferedReader br = new BufferedReader(isr);
	                  
	                String lineContent = null;    
	                while((lineContent = br.readLine())!=null){  
	                	
	                    System.out.println(lineContent);
	                    Tensor input=transformer( lineContent, map );
	                    System.out.println(input);
	                    //def readAllBytesOrExit(path: Path): Array[Byte] = Files.readAllBytes(path)
	                    //Files.write(Paths.get(modelDir, modelName), myGraph.toGraphDef)
	                    byte[] graphDef = readAllBytesOrExit(Paths.get(
	                    		"d:/a",
	                    		"cnn_model_new.h5.pb"));
	                    		
	                    int[] outPuts = new int[35];//结果分类 	
	                    Graph g = new Graph();
	                    g.importGraphDef(graphDef);
	                    
	       			     //System.out.println(input);
	       			     
	       			  try (Session s = new Session(g);
	       					
	   			          // Tensor output = s.runner().feed("main_input1:0",input).fetch("y_output/Softmax").run().get(0)) {
	       					Tensor output = s.runner().feed("main_input1:0",input).fetch("y_output/Softmax").run().get(0)) {
	       				 //Shape shape = output.shape();                
	       				  long[] rshape = output.shape();
	       		          int nlabels = (int) rshape[1];//类别数量
	       		         float[][] copy = new float[1][nlabels];
	       		         output .copyTo(copy);
	       		         
	       		         //获取最大值的索引值
	       		         int maxInx=0;
	       		         
	       		    
	       	        float max=0;
	       	     int jjs=0;//获取索引值
	       	    
	       	     for (int i = 0; i < copy.length; i++) {  
	      	            for (int d = 0; d < copy[i].length; d++) { 
	      	            	if (max <copy[i][d]){
	      	            		max=copy[i][d];
	      	            		jjs=d;//保存最大值的索引
	      	            		 
	      	            	}
	      	            	
	      	            }  
	      	           
	      	        } 
	       	    System.out.println(jjs);  
	   			      }
	                }    
	                br.close();    
	                  
	            } catch (FileNotFoundException e) {    
	                System.out.println("no this file");    
	                e.printStackTrace();    
	            } catch (IOException e) {    
	                System.out.println("io exception");    
	                e.printStackTrace();    
	            }    
	            
		 
		
		 
		  }
		}
