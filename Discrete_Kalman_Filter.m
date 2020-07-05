%EGR 680:Advanced Controls: Project 3- Part2
%Name: Prashant Adhikari
%Professor: Dr. Nicholas Baine
%Objective: To apply Kalman filter to a non-linear system
%process to derive an optimal estimate.
%-----------------------------------------------------------------------------------------------------
load('Project3Part2.mat'); %Importing True_x -> truth measurement and y -> given measurement
%-----------------------------------------------------------------------------------------------------
% Given
r1 = 10;
r2 = 28;
r3 = 8/3;
delta_t = 0.01;
b = [0,0,0.5]';
d = 0.065;
%-----------------------------------------------------------------------------------------------------
%Initialization
X_predict = [0,0,0]'; 
P_predict = 0.35*eye(3); 
X_update=[];
P_update=[];
%-----------------------------------------------------------------------------------------------------
%Using MATLAB inbuilt function handle to compute f and h
f = @(x)([r1*(-x(1)+x(2)),r2*x(1)-x(2)-x(1)*x(3),-r3*x(3)+x(1)*x(2)]');
h = @(x)(sqrt((x(1)-0.5)^2+x(2)^2+x(3)^2));
%-----------------------------------------------------------------------------------------------------
%Jacobian matrix for f and h has been computed manually and the final
%equation is used to make the compilation faster. The mathemtaical derivation for Jacobian equation has
%been derived in report.
Jacobian_f = @(x)([1-delta_t*r1,delta_t*r1*1,0;
             delta_t*(r2-x(3)),1-delta_t*1,-delta_t*x(1);
              delta_t*x(2),delta_t*x(1),1-delta_t*r3]) ;   

Jacobian_h = @(x)((delta_t /sqrt((x(1)-0.5)^2 + x(2)^2 + x(3)^2))*[(x(1)-0.5) x(2) x(3)]);  
%-----------------------------------------------------------------------------------------------------
Q = b*delta_t*b'; %covariance of process noise (wk)
R = d*delta_t*d'; %covariance of measurement/observation noise (vk)
%-----------------------------------------------------------------------------------------------------
%Implementing discrete Kalman Filter
for n = 1:length(True_x)
    X_predict = X_predict + delta_t*f(X_predict);%predicted (a priori) state estimate 
    F = Jacobian_f(X_predict); 
    H = Jacobian_h(X_predict);
    P_predict = F*P_predict*F'+Q; %predicted (a priori) estimate of error covariance
    V = y(:,n) - delta_t*h(X_predict);% innovation (measurement residual)
    Sk = H*P_predict*H'+R; %innovation/measurement covariance    
    K = P_predict*H'*(Sk)^-1; %optimal Kalman gain
    P_predict = (eye(3)-K*H)*P_predict; %updated (a posteriori) estimate at state estimate covariance
    X_predict = X_predict + K*V; %updated (a  posteriori) state estimate                
    traceP(n) =sqrt(trace(P_predict));% trace of estimated error covariance matrix
    X_update=[X_update,X_predict];
    P_update=[P_update,P_predict];

end

%-----------------------------------------------------------------------------------------------------
%plotting true and estimate graphs for state 1, state 2 and state 3
n1=1:1:length(True_x);
subplot(3,2,1);
plot(n1,True_x(1,:),'Linewidth',2);
hold on;
plot(n1,X_update(1,:),'Linewidth',2);
hold on;
grid on;
legend('True', 'Estimate');
title('Estimate values vs True values for state 1');
xlabel('n');
ylabel('Value');


subplot(3,2,2);
plot(n1,True_x(2,:),'Linewidth',2);
hold on;
plot(n1,X_update(2,:),'Linewidth',2);
hold on;
grid on;
legend('True', 'Estimate');
title('Estimate values vs True values for state 2');
xlabel('n');
ylabel('Value');

subplot(3,2,3);
plot(n1,True_x(3,:),'Linewidth',2);
hold on;
plot(n1,X_update(3,:),'Linewidth',2);
hold on;
grid on;
legend('True', 'Estimate');
title('Estimate values vs True values for state 3');
xlabel('n');
ylabel('Value');
%-----------------------------------------------------------------------------------------------------
% Root mean square error calculation for state 1, state 2 and state 3
for i = 1:length(True_x)
    RMSE_State1(i) = sqrt(sum((X_update(1,1:i)-True_x(1,1:i)).^2))/i;
end

for i = 1:length(True_x)
    RMSE_State2(i) = sqrt(sum((X_update(2,1:i)-True_x(2,1:i)).^2))/i;
end

for i = 1:length(True_x)
    RMSE_State3(i) = sqrt(sum((X_update(3,1:i)-True_x(3,1:i)).^2))/i;
end
%-----------------------------------------------------------------------------------------------------
%plotting root mean square error graphs for state 1,state 2 and state 3
subplot(3,2,4);
plot(n1,RMSE_State1,'Linewidth',2);
hold on;
plot(n1,RMSE_State2,'Linewidth',2);
hold on;
plot(n1,RMSE_State3,'Linewidth',2);
grid on;
legend('State 1', 'State 2','State 3');
title('Root mean square error');
xlabel('n');
ylabel('RMSE');
%-----------------------------------------------------------------------------------------------------
%RMSE = sqrt(sum(error,3)/400); %Root mean square error values for state
%1, state 2 and state 3
%-----------------------------------------------------------------------------------------------------
% Plotting trace of estimate error covariance matrix
subplot(3,2,5);
plot(n1,traceP);
title('Trace of estimated error covariance matrix');
xlabel('n');
ylabel('trace(P)');
