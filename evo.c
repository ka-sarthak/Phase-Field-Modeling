#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<complex.h>
#include<fftw3.h>

void main()
{
	fftw_init_threads();

	int i, j, k, i1, j1, k1, l1, nx=512, ny=512;
	double vol=0.2, c0b, c0c;
	
	
	int t=0, nt, t_total=100000;
	double Kbb, Kcc, Kbc, Ka=4.0, Kb=4.0, Kc=4.0;
	double kx, ky, dx=1.0, dy=1.0, dt=0.1, ki=3.2;
	double L1, L2, L3, L4, R1, R2, det;
	double Mbb, Mcc, Mbc, Ma=1.0, Mb=1.0, Mc=1.0;
	double sigma, f, fct, potA, potB, potC, gradA, gradB, gradC, Tgrad, ca, cb, cc;
	char out[40];
	double Kpow4, Kpow2;
	FILE *fp;
	double *C1, *C2, *E0, dummy;
	double misfit=0.005, gammaB=0.01, gammaC=0.0001; 
	double C[2][2][2][2], C11=250, C12=150, C44=100, Ginv11, Ginv12, Ginv21, Ginv22, G[2][2], Ginvdet;
	double kxy[2], sigmaT[2][2], e0[2][2], ro[2][2], perstrreal[2][2];
	
	
	double alphaA=0.865371, alphaB=0.067315, alphaC=0.067315;
	double betaA=0.067315, betaB=0.865371, betaC=0.067315;
	double ss=0.126008842;	

	nt=t_total*10;
	
	C1=(double*)malloc(sizeof(double)*nx*ny);
	C2=(double*)malloc(sizeof(double)*nx*ny);
	E0=(double*)malloc(sizeof(double)*nx*ny);
//inputing the initial condition
	
	fp=fopen("/home/kapster/elastic/onepptelas/191010/criticalrad/with/outB7900.txt", "r");
	for(i=0; i<nx; i++)
	for(j=0; j<ny; j++)
		fscanf(fp, "%lf %lf %lf ", &dummy, &dummy, &C1[j+i*ny]);
	fclose(fp);

	fp=fopen("/home/kapster/elastic/onepptelas/191010/criticalrad/with/outC7900.txt", "r");
	for(i=0; i<nx; i++)
	for(j=0; j<ny; j++)
		fscanf(fp, "%lf %lf %lf ", &dummy, &dummy, &C2[j+i*ny]);
	fclose(fp);
	
//setting the effective Kappas value
	Kbb=Ka+Kb;
	Kcc=Ka+Kc;
	Kbc=Ka;
		
//average composition at fixed volume fraction of beta=0.2 and alpha=0.8
	c0b=vol*(betaB-(alphaB+ss))+(alphaB+ss);		//0.226926
	c0c=vol*(betaC-alphaC)+alphaC;		//0.067315
	
	Mbb= Mb*pow(1-c0b,2) + (Ma+Mc)*pow(c0b,2);
	Mcc= Mc*pow(1-c0c,2) + (Ma+Mb)*pow(c0c,2);
	Mbc= Mb*c0c*(1-c0b) + Mc*c0b*(1-c0c) - Ma*c0b*c0c;
	
//defining elasticity tensor

	for(i1=0; i1<2; i1++)
		for(j1=0; j1<2; j1++)
			for(k1=0; k1<2; k1++)
				for(l1=0; l1<2; l1++)
					C[i1][j1][k1][l1]=0;
	C[0][0][0][0]=C11;
	C[1][1][1][1]=C11;
	C[0][0][1][1]=C12;
	C[1][1][0][0]=C12;
	C[0][1][0][1]=C44;
	C[1][0][1][0]=C44;
	C[0][1][1][0]=C44;
	C[1][0][0][1]=C44;	
	
//defining eigen strain matrix, i.e, sigmaT

	sigmaT[0][0]=misfit*(C11+C12);
	sigmaT[0][1]=0;
	sigmaT[1][0]=0;
	sigmaT[1][1]=misfit*(C11+C12);

//kronecker 
	ro[0][0]=1;
	ro[0][1]=0;
	ro[1][0]=0;
	ro[1][1]=1;

//defining complex space 
	fftw_complex *c1, *c2, *g1, *g2,  *beta1, *beta2, *beta;
	fftw_complex *u1, *u2, *perstr11, *perstr12, *perstr21, *perstr22;
	fftw_plan forward, backward;


	c1=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	c2=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	
	g1=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	g2=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	
	beta=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	beta1=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	beta2=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	


	u1=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	u2=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	
	perstr11=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	perstr12=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	perstr21=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);
	perstr22=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*ny);

	fftw_plan_with_nthreads(4);
	
	forward=fftw_plan_dft_2d(nx, ny, c1, c1, FFTW_FORWARD, FFTW_ESTIMATE);
	backward=fftw_plan_dft_2d(nx, ny, c1, c1, FFTW_BACKWARD, FFTW_ESTIMATE);

	
///////////////////////////////////////////////////////// elasticity //////////////////////////


//getting composition into frequency space
	for(i=0; i<nx; i++)
	for(j=0; j<ny; j++)
	{
		k=j+i*ny;
		
		c1[k]= C1[k]+0.0*_Complex_I;
		c2[k]= C2[k]+0.0*_Complex_I;
		g1[k]=ki*(1-2*c1[k]-c2[k]) + log(c1[k]/(1-c1[k]-c2[k]));
		g2[k]=ki*(1-c1[k]-2*c2[k]) + log(c2[k]/(1-c1[k]-c2[k]));

		beta[k]=(c1[k]-alphaB)*gammaB+(c2[k]-alphaC)*gammaC;
		beta1[k]=gammaB+0.0*_Complex_I;
		beta2[k]=gammaC+0.0*_Complex_I;
		
		u1[k]=0+0.0*_Complex_I;
		u2[k]=0+0.0*_Complex_I;
		perstr11[k]=0+0.0*_Complex_I;
		perstr12[k]=0+0.0*_Complex_I;
		perstr21[k]=0+0.0*_Complex_I;
		perstr22[k]=0+0.0*_Complex_I;
	}

	

	fftw_execute_dft(forward, c1, c1);
	fftw_execute_dft(forward, g1, g1);
	fftw_execute_dft(forward, c2, c2);
	fftw_execute_dft(forward, g2, g2);
	fftw_execute_dft(forward,beta,beta);
	
	//because beta1, beta2 are constant functions, no need to update them after every time step (applicable only for the current beta function, ie, (cb-cb0)gammaB+(cc-cc0)gammaC
	fftw_execute_dft(forward,beta1,beta1);
	fftw_execute_dft(forward,beta2,beta2);
				
							
	fftw_execute_dft(forward,u1,u1);
	fftw_execute_dft(forward,u2,u2);
	fftw_execute_dft(forward,perstr11,perstr11);
	fftw_execute_dft(forward,perstr12,perstr12);
	fftw_execute_dft(forward,perstr21,perstr21);
	fftw_execute_dft(forward,perstr22,perstr22);



	for(t=79001; t<=t_total; t++)
	{	
		if(t%1000==0)
		printf("Last T reached=%d\n", t/10);
	//solving the mech equi for entire space, finding all the strains 
		for(i=1; i<nx-1; i++)
		{	
			
			if(i<(int)(nx/2))	kx=2.0*M_PI*i/(dx*nx);
			else			kx=2.0*M_PI*(nx-i)/(dx*nx);	

			for(j=1; j<ny-1; j++)
			{
				k=j+i*ny;
				
				if(j<ny/2)	ky=2.0*M_PI*j/(dy*ny);
				else		ky=2.0*M_PI*(ny-j)/(dy*ny);
				
		//getting G tensor which is (Cijkl*gj*gk)^(-1)
				
				Ginv11= C11*kx*kx+C44*ky*ky;
				Ginv12= (C12+C44)*kx*ky;
				Ginv21= Ginv12;
				Ginv22= C44*kx*kx+C11*ky*ky;
				
				Ginvdet=Ginv11*Ginv22-Ginv21*Ginv12;
				
				G[0][0]=  Ginv22/Ginvdet;
				G[0][1]= -Ginv12/Ginvdet;
				G[1][0]= -Ginv21/Ginvdet;
				G[1][1]=  Ginv11/Ginvdet;
				
								
		//solving for mechanical equilibrium to get displacement at given point
				
				u1[k]=0+0.0*_Complex_I;
				u2[k]=0+0.0*_Complex_I;
				
				kxy[0]=kx;
				kxy[1]=ky;
				//take ft of beta function				
				for(i1=0; i1<2; i1++)
				for(j1=0; j1<2; j1++)
					u1[k]+=-_Complex_I*G[i1][0]*sigmaT[i1][j1]*kxy[j1]*beta[k];
				
				for(i1=0; i1<2; i1++)
				for(j1=0; j1<2; j1++)
					u2[k]+=-_Complex_I*G[i1][1]*sigmaT[i1][j1]*kxy[j1]*beta[k];



		//determination of periodic strain at given point
				perstr11[k]=-_Complex_I*kx*u1[k];
				perstr12[k]=-_Complex_I*ky*u1[k];
				perstr21[k]=-_Complex_I*kx*u2[k];
				perstr22[k]=-_Complex_I*ky*u2[k];

		//determination of eigen strain at given point
				
			//	fftw_execute_dft(backward,beta,beta);
				
				//eigen strain
				//for(i1=0; i1<2; i1++)
				//for(j1=0; j1<2; j1++)
				//e0[i1][j1]=misfit*(beta[k]/(double)(nx*ny))*ro[i1][j1];

			//	fftw_execute_dft(forward,beta,beta);
				
			}
		}	
	
	//getting periodic strain field into real space
		fftw_execute_dft(backward,perstr11,perstr11);
		fftw_execute_dft(backward,perstr12,perstr12);
		fftw_execute_dft(backward,perstr21,perstr21);
		fftw_execute_dft(backward,perstr22,perstr22);
		
		fftw_execute_dft(backward,beta,beta);
	
	//calculating elastic energy landscape
		for(i=0; i<nx; i++)
		{	
			
			if(i<(int)(nx/2))	kx=2.0*M_PI*i/(dx*nx);
			else			kx=2.0*M_PI*(nx-i)/(dx*nx);	

			for(j=0; j<ny; j++)
			{
				k=j+i*ny;
				
				if(j<ny/2)	ky=2.0*M_PI*j/(dy*ny);
				else		ky=2.0*M_PI*(ny-j)/(dy*ny);
				
				perstrreal[0][0]=creal(perstr11[k]);
				perstrreal[0][1]=creal(perstr12[k]);
				perstrreal[1][0]=creal(perstr21[k]);
				perstrreal[1][1]=creal(perstr22[k]);
				
				perstr11[k]=0;
				perstr12[k]=0;
				perstr21[k]=0;	
				perstr22[k]=0;
					
				//eigen strain
				for(i1=0; i1<2; i1++)
				for(j1=0; j1<2; j1++)
				e0[i1][j1]=misfit*(beta[k]/(double)(nx*ny))*ro[i1][j1];

				//determination of E0
				E0[k]=0;
				for(i1=0; i1<2; i1++)
				for(j1=0; j1<2; j1++)
				for(k1=0; k1<2; k1++)
				for(l1=0; l1<2; l1++)
					E0[k]-=C[i1][j1][k1][l1]*(perstrreal[k1][l1]-e0[k1][l1])*misfit*ro[i1][j1];	
			}
		}		
				
//evolution of the system

		for(i=0; i<nx; i++)
		{	
			
			if(i<(int)(nx/2))	kx=2.0*M_PI*i/(dx*nx);
			else			kx=2.0*M_PI*(nx-i)/(dx*nx);	

			for(j=0; j<ny; j++)
			{
				k=j+i*ny;
				
				if(j<ny/2)	ky=2.0*M_PI*j/(dy*ny);
				else		ky=2.0*M_PI*(ny-j)/(dy*ny);
					

	
				//evolution	
				Kpow4=(pow(kx,4)+pow(ky,4)+2*pow(kx,2)*pow(ky,2));
				Kpow2=(pow(kx,2)+pow(ky,2));
				
				L1= 1 + 2*(Mbb*Kbb-Mbc*Kbc)*Kpow4*dt;
				L2= 2*Kpow4*dt*(Mbb*Kbc-Mbc*Kcc);
				L3= 2*Kpow4*dt*(Mcc*Kbc-Mbc*Kbb);
				L4= 1 + 2*(Mcc*Kcc-Mbc*Kbc)*Kpow4*dt;
			
				R1= c1[k] - Kpow2*dt*( Mbb*(g1[k]+beta1[k]*E0[k])- Mbc*(g2[k]+beta2[k]*E0[k]) );
				R2= c2[k] - Kpow2*dt*( Mcc*(g2[k]+beta2[k]*E0[k])- Mbc*(g1[k]+beta1[k]*E0[k]) );
			
				det= L1*L4 - L2*L3;
			
				c1[k]= (R1*L4 - R2*L2)/det;
				c2[k]= (L1*R2 - L3*R1)/det;
			
			}
		}	
		
		fftw_execute_dft(backward, c1, c1);
		fftw_execute_dft(backward, c2, c2);
		
		

//updation of c1 and g1 in time space	
		
		for(i=0; i<nx; i++)
		for(j=0; j<ny; j++)
		{
			k=j+i*ny;
		
			c1[k]= ((creal(c1[k]))/(double)(nx*ny))+0.0*_Complex_I;
			c2[k]= ((creal(c2[k]))/(double)(nx*ny))+0.0*_Complex_I;
			g1[k]=ki*(1-2*c1[k]-c2[k]) + log(c1[k]/(1-c1[k]-c2[k]));
			g2[k]=ki*(1-c1[k]-2*c2[k]) + log(c2[k]/(1-c1[k]-c2[k]));
			beta[k]=((c1[k]-alphaB)*gammaB+(c2[k]-alphaC)*gammaC);
		}
		
		if(t%1000==0)
		{
		/* 	printf("E0=%f\n", E0);
			printf("per1=%f\n", perstrreal[0][0]);
			printf("per2=%f\n", perstrreal[0][1]);
			printf("per3=%f\n", perstrreal[1][0]);
			printf("per4=%f\n", perstrreal[1][1]);
		*/		

			sprintf(out, "G1%d.txt", t/10);		
			fp=fopen(out, "w");
		
			for(i=0; i<nx; i++)
			{	for(j=0; j<ny; j++)
				fprintf(fp, "%f\t%f\t%lf\n", i*dx, j*dy, creal(g1[j+i*ny]));
				fprintf(fp, "\n");
			}
			fclose(fp);

			sprintf(out, "E%d.txt", t/10);		
			fp=fopen(out, "w");
		
			for(i=0; i<nx; i++)
			{	for(j=0; j<ny; j++)
				fprintf(fp, "%f\t%f\t%lf\n", i*dx, j*dy, E0[j+i*ny]);
				fprintf(fp, "\n");
			}
			fclose(fp);

			//printing data file of b
			sprintf(out, "outB%d.txt", t/10);		
			fp=fopen(out, "w");
		
			for(i=0; i<nx; i++)
			{	for(j=0; j<ny; j++)
				fprintf(fp, "%f\t%f\t%lf\n", i*dx, j*dy, creal(c1[j+i*ny]));
				fprintf(fp, "\n");
			}
			fclose(fp);

			sprintf(out, "outC%d.txt", t/10);		
			fp=fopen(out, "w");
		
			for(i=0; i<nx; i++)
			{	for(j=0; j<ny; j++)
				fprintf(fp, "%f\t%f\t%lf\n", i*dx, j*dy, creal(c2[j+i*ny]));
				fprintf(fp, "\n");
			}
			fclose(fp);
			
		}
		
				
		fftw_execute_dft(forward, c1, c1);
		fftw_execute_dft(forward, g1, g1);
		fftw_execute_dft(forward, c2, c2);
		fftw_execute_dft(forward, g2, g2);
		fftw_execute_dft(forward, beta, beta);
		fftw_execute_dft(forward,perstr11,perstr11);
		fftw_execute_dft(forward,perstr12,perstr12);
		fftw_execute_dft(forward,perstr21,perstr21);
		fftw_execute_dft(forward,perstr22,perstr22);
		
		

	}
	
	
	fftw_free(c1);
	fftw_free(g1);
	fftw_free(c2);
	fftw_free(g2);
	fftw_free(u1);
	fftw_free(u2);
	fftw_free(beta1);
	fftw_free(beta2);
	fftw_free(perstr11);
	fftw_free(perstr12);
	fftw_free(perstr21);
	fftw_free(perstr22);
	fftw_free(beta);

	fftw_destroy_plan(forward);
	fftw_destroy_plan(backward);
	fftw_cleanup_threads();
	fftw_cleanup();

	free(C1);
	free(C2);
	
			
}
