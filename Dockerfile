# pull the official base image  
FROM node:13.12.0-alpine as dev

FROM nginxinc/nginx-unprivileged
COPY . /etc/nginx/html

USER 101
COPY nginx.conf /etc/nginx/conf.d/default.conf
CMD ["nginx-debug", "-g", "daemon off;"]