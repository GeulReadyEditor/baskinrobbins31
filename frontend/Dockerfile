FROM node:12-alpine AS builder
WORKDIR /app
COPY package.json ./
RUN yarn install
COPY ./ ./
RUN yarn run build

FROM nginx
EXPOSE 3000
COPY ./nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html