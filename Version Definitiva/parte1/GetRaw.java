package parte1;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class GetRaw 
{
	private String ruta;
	
	public GetRaw( String pRuta ) { this.ruta = pRuta; }
	
	// CONVIERTE A FORMATO ARFF EL FICHERO DE REGUNTAS EN FORMATO CSV.
	public void getRawARFF( String pNombre ) 
	{
		String datoscsv = this.ruta + "/" + pNombre + ".csv"; 
		String rawarff = this.ruta + "/" + pNombre + "_RaW.arff";
		String tmparf = this.ruta + "/" + pNombre + "_RaW_SinComillas.arff";
		String tmp = ruta + "/ztmp.csv";
	    	
    	try
        {
        	System.out.println("Iniciando conversion del fichero " + pNombre + "...");
	        	
        	BufferedReader br = new BufferedReader(new FileReader( datoscsv ));
        	BufferedWriter bw = new BufferedWriter(new FileWriter( tmp ));
				
        	String linea;
        	while((linea = br.readLine()) != null)
        	{
        		// ELIMINAR CUALQUIER CARACTER NO ASCII DE LAS INSTANCIAS
        		linea = linea.replaceAll("[^\\p{ASCII}]", "");
	        	linea = linea.replaceAll("\\.", "");
        		linea = linea.replaceAll("\\?", "");
        		linea = linea.replaceAll("\\:", "");
        		linea = linea.replaceAll("\\-", "");
        		linea = linea.replaceAll("\\&", "");
        		linea = linea.replaceAll("'", "");
        		linea = linea.replaceAll("`", "");
	        			
        		// DIVIDIR EN TRES PARTES PARA SEPARAR LOS ATRIBUTOS
        		String[] partes = linea.split(",", 3);
	        		
        		// ESCRIBIR EL RESULTADO
        		bw.write(partes[2]);
        		bw.write(" , ");
        		bw.write(partes[1]);
        		
                bw.newLine();		
        	}
        	br.close();
        	bw.close();
        	System.out.println("   Fichero temporal limpio...");
	        	
        	// CARGAR EL CSV LIMPIO Y MARCAR COMO STRING LA PREGUNTA
           	CSVLoader loader = new CSVLoader();
            loader.setSource( new File( tmp ) );
            loader.setStringAttributes("last");
        	Instances data = loader.getDataSet();    	
        	System.out.println("   Fichero temporal cargado...");
	        	
        	// SETTEAR NOMBRE DE LA RELACION
        	data.setRelationName("preguntas_" + pNombre);
        	System.out.println("   Nombre de la relacion establecida...");
	        	
        	// GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
        	BufferedWriter writer = new BufferedWriter(new FileWriter( rawarff ));
        	writer.write(data.toString());
        	writer.flush();
        	writer.close();
        	System.out.println("Conversion del fichero " + pNombre + " finalizada... \n");
	        	 
        	// QUITAR COMILLAS
        	BufferedReader read = new BufferedReader(new FileReader( rawarff ));
        	BufferedWriter write = new BufferedWriter(new FileWriter( tmparf ));
			
        	String line;
        	int encabezado = 0;
        	while( (line = read.readLine()) != null )
        	{
        		encabezado++;
        		if(encabezado < 5) 
        		{
        			line = line.replaceAll("'", "");
            		line = line.replaceAll("`", "");
        		}
        		write.write(line);
        		write.newLine();
        	}
        	write.close();
        }
        catch(Exception e) { e.printStackTrace(); }
	}	
}
